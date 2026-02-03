import yaml
import os
import subprocess
import shutil
import glob
import logging
import re # Import regex for parsing filenames
import random 
from pathlib import Path
import sys

# --- Configuration ---
CONFIG_FILE = 'pl_config.yaml'

# --- Logging Setup ---
# Setup basic logging configuration
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)]) # Log to stdout

# --- Helper Functions ---

def run_command(command_list, cwd=None):
    """
    Runs a command using subprocess, allowing its stdout/stderr to stream directly.
    Logs the command being run and reports success or failure based on exit code.
    """
    logging.info(f"Running command: {' '.join(command_list)}")
    try:
        process = subprocess.run(
            command_list,
            check=True,
            text=True,
            cwd=cwd,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        # Note: Command success logging happens below after check=True passes
        logging.info(f"Command finished successfully: {' '.join(command_list)}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with exit code {e.returncode}: {' '.join(command_list)}")
        # stderr is already streamed
        return False
    except FileNotFoundError:
        logging.error(f"Command not found: {command_list[0]}. Check path or script name for command: {' '.join(command_list)}")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred running command: {' '.join(command_list)}")
        logging.error(f"Error details: {e}")
        return False

def select_best_checkpoint_after_epoch(checkpoint_dir, min_epoch_for_selection):
    """
    Selects the best checkpoint based on val_loss from those saved
    after a specified minimum epoch. Falls back to last.ckpt if none meet criteria.

    Args:
        checkpoint_dir (str or Path): Directory containing checkpoints.
        min_epoch_for_selection (int): Minimum epoch number (inclusive) to consider.

    Returns:
        Path or None: Path object to the selected checkpoint, or None if none found.
    """
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.is_dir():
        logging.error(f"Checkpoint directory not found: {checkpoint_path}")
        return None

    all_ckpts = list(checkpoint_path.glob('*.ckpt'))
    last_ckpt_path = checkpoint_path / 'last.ckpt'
    epoch_ckpts = [p for p in all_ckpts if p != last_ckpt_path]

    logging.info(f"Found {len(epoch_ckpts)} specific epoch checkpoints and {'last.ckpt' if last_ckpt_path.exists() else 'no last.ckpt'}.")

    considered_ckpts = []
    for ckpt_path in epoch_ckpts:
        match_epoch = re.search(r'epoch=(\d+)', ckpt_path.stem)
        match_loss = re.search(r'val_loss=([\d\.]+)', ckpt_path.stem)

        if match_epoch and match_loss:
            try:
                epoch = int(match_epoch.group(1))
                val_loss = float(match_loss.group(1))
                # Use '>=' to include the min_epoch itself
                if epoch >= min_epoch_for_selection:
                    considered_ckpts.append((val_loss, epoch, ckpt_path))
                    logging.debug(f"Considering ckpt: {ckpt_path.name} (epoch={epoch}, loss={val_loss:.4f})")
                else:
                     logging.debug(f"Skipping ckpt (epoch < {min_epoch_for_selection}): {ckpt_path.name} (epoch={epoch})")
            except ValueError:
                logging.warning(f"Could not parse epoch/loss from filename: {ckpt_path.name}")
            except Exception as e:
                 logging.warning(f"Error parsing file {ckpt_path.name}: {e}")
        else:
            logging.warning(f"Could not extract epoch or val_loss from filename: {ckpt_path.name}")

    best_ckpt_path = None
    if considered_ckpts:
        considered_ckpts.sort(key=lambda x: (x[0], -x[1])) # Sort by loss asc, epoch desc
        best_loss, best_epoch, best_ckpt_path = considered_ckpts[0]
        logging.info(f"Selected best checkpoint from epoch >= {min_epoch_for_selection}: {best_ckpt_path.name} (epoch={best_epoch}, val_loss={best_loss:.4f})")
        return best_ckpt_path # Return the path to the best specific epoch checkpoint
    else:
        logging.warning(f"No checkpoints found with epoch >= {min_epoch_for_selection}. Falling back to last.ckpt.")
        if last_ckpt_path.exists():
            logging.info(f"Using last.ckpt: {last_ckpt_path}")
            return last_ckpt_path # Return the path to last.ckpt
        else:
            logging.error("Fallback failed: last.ckpt does not exist.")
            # Final fallback attempt: Find best overall loss among any epoch checkpoints
            if epoch_ckpts:
                 logging.warning("Attempting to find best overall checkpoint regardless of epoch as final fallback...")
                 best_loss = float('inf')
                 fallback_path = None
                 for ckpt_path in epoch_ckpts:
                     match_loss = re.search(r'val_loss=([\d\.]+)', ckpt_path.stem)
                     if match_loss:
                         try:
                             loss = float(match_loss.group(1))
                             if loss < best_loss:
                                 best_loss = loss
                                 fallback_path = ckpt_path
                         except ValueError: continue
                 if fallback_path:
                     logging.warning(f"Using best overall checkpoint as final fallback: {fallback_path.name} (loss={best_loss:.4f})")
                     return fallback_path

            logging.error("No suitable checkpoint could be found.")
            return None

def copy_directory_contents(src_dir, dest_dir):
    """Copies all files and subdirectories from src_dir into dest_dir."""
    src_path = Path(src_dir)
    dest_path = Path(dest_dir)
    if not src_path.is_dir():
        logging.error(f"Source for copy is not a directory: {src_path}")
        return False
    dest_path.mkdir(parents=True, exist_ok=True)
    copy_errors = 0
    try:
        for item in src_path.iterdir():
            dest_item = dest_path / item.name
            try:
                if item.is_dir():
                    shutil.copytree(item, dest_item, dirs_exist_ok=True)
                    logging.debug(f"Copied directory {item} to {dest_item}")
                else:
                    shutil.copy2(item, dest_item) # copy2 preserves metadata
                    logging.debug(f"Copied file {item} to {dest_item}")
            except Exception as item_e:
                 logging.error(f"Error copying item {item} to {dest_item}: {item_e}")
                 copy_errors += 1
        if copy_errors == 0:
            logging.info(f"Successfully copied contents of {src_path} to {dest_path}")
            return True
        else:
            logging.error(f"Encountered {copy_errors} errors while copying from {src_path}")
            return False # Indicate partial or failed copy
    except Exception as e:
        logging.error(f"Error preparing to copy contents from {src_path} to {dest_path}: {e}")
        return False

# --- Main Pipeline ---
def main():
    logging.info("Starting Pseudo-Labeling Pipeline...")

    # 1. Load Configuration
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Loaded configuration from {CONFIG_FILE}")
    except FileNotFoundError:
        logging.error(f"Configuration file {CONFIG_FILE} not found.")
        return
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        return

    # --- Resolve Paths ---
    base_output_dir = Path(config['base_output_dir'])
    initial_checkpoint = Path(config['initial_checkpoint'])
    original_train_dir = Path(config['original_train_dir'])
    unlabeled_lmdb_path = Path(config['unlabeled_lmdb_path'])
    unlabeled_images_root = Path(config['unlabeled_images_root'])
    original_val_dir = Path(config['original_val_dir'])
    original_test_dir = Path(config['original_test_dir'])
    test_lmdb_foldername = config['test_lmdb_foldername']

    # --- Script Paths ---
    save_script = config['save_high_confidence_script']
    convert_script = config['convert_lmdb_script']
    train_script = config['train_script']
    test_script = config['test_script']

    # --- Get Configurable threshold ---
    min_epoch_for_selection = config.get('min_epoch_for_selection', 10)
    logging.info(f"Will select best checkpoint from epochs >= {min_epoch_for_selection}")

    # --- Initialize ---
    current_checkpoint_path = initial_checkpoint # Path to the checkpoint file used for training/inference
    num_cycles = config['num_cycles']
    prev_cycle_data_root = None

    # --- Start Cycles ---
    for cycle in range(num_cycles):
        cycle_step_failed = False # Flag to track if any step in the cycle fails
        logging.info(f"--- Starting Cycle {cycle + 1}/{num_cycles} ---")
        cycle_output_dir = base_output_dir / f"cycle_{cycle}"
        cycle_output_dir.mkdir(parents=True, exist_ok=True)

        # --- Cleanup previous cycle's data ---
        if prev_cycle_data_root and config.get('cleanup_intermediate', True): # Default cleanup to True
             if prev_cycle_data_root.exists():
                 logging.info(f"Cleaning up previous cycle's combined data directory: {prev_cycle_data_root}")
                 try: shutil.rmtree(prev_cycle_data_root)
                 except Exception as e: logging.error(f"Failed to remove previous cycle data {prev_cycle_data_root}: {e}")
             else: logging.warning(f"Previous cycle data directory not found for cleanup: {prev_cycle_data_root}")

        # Define paths for this cycle
        high_conf_output_dir = cycle_output_dir / "high_confidence_temp"
        pseudo_lmdb_dir = cycle_output_dir / f"pseudo_lmdb_cycle_{cycle}"
        cycle_data_root = cycle_output_dir / "data_for_training"
        model_name = config.get('train_overrides', {}).get('model', 'model')
        train_output_dir = cycle_output_dir / f"training_output_{model_name}"

        # --- Step 1: Get High-Confidence Predictions ---
        logging.info(f"[Cycle {cycle + 1}] Step 1: Generating high-confidence predictions...")
        if not cycle_step_failed:
            high_conf_output_dir.mkdir(parents=True, exist_ok=True)
            cmd_infer = [
                'python', save_script,
                '--checkpoint', str(current_checkpoint_path), # Use current checkpoint
                '--data_root', str(unlabeled_lmdb_path.parent),
                '--test_folder', unlabeled_lmdb_path.name,
                '--image_root', str(unlabeled_images_root),
                '--output_path', str(high_conf_output_dir),
                '--confidence_threshold', str(config['confidence_threshold']),
                '--batch_size', str(config.get('inference_batch_size', 128)),
                '--num_workers', str(config.get('num_workers', 8)),
                '--device', str(config.get('device', 'cuda'))
            ]
            if not run_command(cmd_infer):
                logging.error(f"[Cycle {cycle + 1}] Failed to generate high-confidence predictions. Skipping rest of cycle.")
                cycle_step_failed = True
            elif not (high_conf_output_dir / "images").exists() or not (high_conf_output_dir / "gt.txt").exists():
                 logging.error(f"[Cycle {cycle + 1}] High confidence gt.txt or images directory missing after inference. Skipping rest of cycle.")
                 cycle_step_failed = True
        
        # === Step 1.5: Sample High confidence data ===
        gt_file_for_lmdb = high_conf_output_dir / "gt.txt"
        images_dir_for_lmdb = high_conf_output_dir / "images"
        
        logging.info(f'[cycle {cycle+1}] Step 1.5: Sampling high-confidence data ')
        if not cycle_step_failed and config['max_pseudo_samples_per_cycle'] is not None:
            original_gt_path = high_conf_output_dir / 'gt.txt'
            image_dir = high_conf_output_dir / 'images'
            sampled_gt_path = high_conf_output_dir/ 'gt_sampled.txt'

            if original_gt_path.exists() and image_dir.is_dir():
                logging.info(f"checking high-confidence sample count against limit set")
                try:
                    with open(original_gt_path, 'r', encoding='utf-8') as f:
                        all_lines = f.readlines()
                        total_found = len(all_lines)
                        logging.info(f'found {total_found} high confidence samples \n\n')
                    
                    if total_found > config['max_pseudo_samples_per_cycle']:
                        logging.info(f"Sampling {config['max_pseudo_samples_per_cycle']} samples from {total_found}")

                        sampled_lines = random.sample(all_lines, config['max_pseudo_samples_per_cycle'])
                        with open(sampled_gt_path, 'w', encoding='utf-8') as f_sampled:
                            f_sampled.writelines(sampled_lines)
                        logging.info(f"Saved sampled ground truth to {sampled_gt_path}")
                        gt_file_for_lmdb = sampled_gt_path                   
                except Exception as e:
                    logging.error(f"Error {e}")

        # --- Step 2: Convert High-Confidence Data to LMDB ---
        logging.info(f"[Cycle {cycle + 1}] Step 2: Converting high-confidence data to LMDB...")
        if not cycle_step_failed:
            if pseudo_lmdb_dir.exists():
                logging.warning(f"Removing existing pseudo LMDB directory: {pseudo_lmdb_dir}")
                shutil.rmtree(pseudo_lmdb_dir)
            cmd_convert = [
                'python', convert_script,
                '--inputPath', str(images_dir_for_lmdb),
                '--gtFile', str(gt_file_for_lmdb),
                '--outputPath', str(pseudo_lmdb_dir),
                '--checkValid'
            ]
            if not run_command(cmd_convert):
                logging.error(f"[Cycle {cycle + 1}] Failed to convert pseudo-labels to LMDB. Skipping rest of cycle.")
                cycle_step_failed = True

            # Cleanup intermediate dir regardless of conversion success if flag is set
            if config.get('cleanup_intermediate', True):
                logging.info(f"Cleaning up intermediate high-confidence files in {high_conf_output_dir}")
                shutil.rmtree(high_conf_output_dir, ignore_errors=True)

        # --- Step 3: Prepare Combined Training Data ---
        logging.info(f"[Cycle {cycle + 1}] Step 3: Preparing combined data directory...")
        if not cycle_step_failed:
            if cycle_data_root.exists():
                logging.warning(f"Removing existing combined data directory: {cycle_data_root}")
                shutil.rmtree(cycle_data_root)
            cycle_train_dir = cycle_data_root / "train/real/"
            cycle_val_dir = cycle_data_root / "val"
            cycle_test_dir = cycle_data_root / "test"
            cycle_train_dir.mkdir(parents=True, exist_ok=True)
            cycle_val_dir.mkdir(parents=True, exist_ok=True)
            cycle_test_dir.mkdir(parents=True, exist_ok=True)

            copy_success = True
            logging.info(f"Copying original train data from {original_train_dir}/* to {cycle_train_dir}")
            if not copy_directory_contents(original_train_dir, cycle_train_dir): copy_success = False
            logging.info(f"Copying pseudo-labeled data {pseudo_lmdb_dir} to {cycle_train_dir / pseudo_lmdb_dir.name}")
            try: shutil.copytree(pseudo_lmdb_dir, cycle_train_dir / pseudo_lmdb_dir.name)
            except Exception as e: logging.error(f"Failed to copy pseudo LMDB {pseudo_lmdb_dir}: {e}"); copy_success = False
            logging.info(f"Copying original validation data from {original_val_dir}/* to {cycle_val_dir}")
            if not copy_directory_contents(original_val_dir, cycle_val_dir): copy_success = False
            logging.info(f"Copying original test data from {original_test_dir}/* to {cycle_test_dir}")
            if not copy_directory_contents(original_test_dir, cycle_test_dir): copy_success = False

            if not copy_success:
                 logging.error(f"[Cycle {cycle + 1}] Failed to prepare combined data directory. Skipping rest of cycle.")
                 cycle_step_failed = True

        # --- Step 4: Train Model ---
        logging.info(f"[Cycle {cycle + 1}] Step 4: Starting training...")
        if not cycle_step_failed:
            hydra_overrides = [
                f"data.root_dir={cycle_data_root}",
                f"hydra.run.dir={train_output_dir}",
                f"pretrained={current_checkpoint_path}", # Use current checkpoint path
                f"ckpt_path=null",
            ]
            # Append overrides from config file
            for key, value in config.get('train_overrides', {}).items():
                 if not key.startswith('data.root'):
                     hydra_overrides.append(f"{key}={value}")
            cmd_train = [
                'python', train_script,
                '--config-path', config['hydra_train_config_path'],
                '--config-name', config['hydra_train_config_name'],
            ] + hydra_overrides
            if not run_command(cmd_train):
                logging.error(f"[Cycle {cycle + 1}] Training failed. Skipping rest of cycle.")
                cycle_step_failed = True


        # --- Step 5: Select and Finalize Best Checkpoint ---
        logging.info(f"[Cycle {cycle + 1}] Step 5: Selecting and finalizing best checkpoint...")
        final_checkpoint_path_for_cycle = None # Path to the definitive checkpoint for this cycle (e.g., best.ckpt)
        checkpoints_dir = None # Define checkpoints_dir outside the try block

        if not cycle_step_failed:
            try:
                # Find the hydra output directory for this training run
                subdirs = [d for d in train_output_dir.iterdir() if d.is_dir()]
                if not subdirs: raise FileNotFoundError(f"No subdirectories found within Hydra run output: {train_output_dir}")
                latest_run_dir = max(subdirs, key=os.path.getmtime)
                checkpoints_dir = latest_run_dir / 'checkpoints'
                if not checkpoints_dir.is_dir():
                     # Check if checkpoints are in the run directory itself (less common for PL)
                     if (latest_run_dir / 'last.ckpt').exists(): checkpoints_dir = latest_run_dir
                     else: raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir} or {latest_run_dir}")

                logging.info(f"Looking for checkpoints in: {checkpoints_dir}")

                # Select the best checkpoint based on criteria (epoch > min_epoch, lowest val_loss)
                selected_ckpt_path = select_best_checkpoint_after_epoch(checkpoints_dir, min_epoch_for_selection)

                if selected_ckpt_path and selected_ckpt_path.exists():
                    # --- Rename/Copy to best.ckpt ---
                    target_best_path = checkpoints_dir / "best.ckpt"
                    logging.info(f"Selected checkpoint: {selected_ckpt_path.name}. Preparing to finalize as {target_best_path.name}")

                    # Remove existing best.ckpt if it exists and is not the same file we selected
                    if target_best_path.exists() and target_best_path.resolve() != selected_ckpt_path.resolve():
                        logging.warning(f"Removing existing {target_best_path.name}")
                        try:
                            target_best_path.unlink()
                        except OSError as e:
                            logging.error(f"Error removing existing {target_best_path.name}: {e}")
                            # Continue, maybe rename/copy will still work or fail gracefully

                    # If selected is last.ckpt, COPY it to best.ckpt
                    if selected_ckpt_path.name == 'last.ckpt':
                        logging.info(f"Selected checkpoint is {selected_ckpt_path.name}. Copying to {target_best_path.name}")
                        try:
                            shutil.copy2(selected_ckpt_path, target_best_path) # copy2 preserves metadata
                            final_checkpoint_path_for_cycle = target_best_path # Use the copied path
                            logging.info(f"Successfully copied {selected_ckpt_path.name} to {target_best_path.name}")
                        except Exception as e:
                            logging.error(f"Failed to copy {selected_ckpt_path.name} to {target_best_path.name}: {e}")
                            final_checkpoint_path_for_cycle = selected_ckpt_path # Fallback to using last.ckpt directly
                    # Otherwise (it's an epoch checkpoint), RENAME it to best.ckpt
                    elif selected_ckpt_path.resolve() != target_best_path.resolve():
                         logging.info(f"Renaming {selected_ckpt_path.name} to {target_best_path.name}")
                         try:
                             selected_ckpt_path.rename(target_best_path)
                             final_checkpoint_path_for_cycle = target_best_path # Use the renamed path
                             logging.info(f"Successfully renamed {selected_ckpt_path.name} to {target_best_path.name}")
                         except OSError as e:
                             logging.error(f"Failed to rename checkpoint {selected_ckpt_path.name} to {target_best_path.name}: {e}")
                             final_checkpoint_path_for_cycle = selected_ckpt_path # Fallback to using original epoch name
                         except Exception as e:
                              logging.error(f"Unexpected error during checkpoint rename: {e}")
                              final_checkpoint_path_for_cycle = selected_ckpt_path # Fallback
                    else:
                        # Selected path is already best.ckpt (shouldn't happen with current select logic, but safe check)
                         logging.info(f"Selected checkpoint {selected_ckpt_path.name} is already the target {target_best_path.name}. No action needed.")
                         final_checkpoint_path_for_cycle = target_best_path
                    # --- End Rename/Copy ---
                else:
                    logging.error(f"[Cycle {cycle + 1}] Could not find any suitable checkpoint after selection.")
                    final_checkpoint_path_for_cycle = current_checkpoint_path # Fallback
                    cycle_step_failed = True # Mark failure if no checkpoint selected

            except FileNotFoundError as e:
                 logging.error(f"[Cycle {cycle + 1}] Error finding checkpoints directory: {e}")
                 final_checkpoint_path_for_cycle = current_checkpoint_path # Fallback
                 cycle_step_failed = True
            except Exception as e:
                 logging.error(f"[Cycle {cycle + 1}] Unexpected error during checkpoint selection/finalization: {e}")
                 final_checkpoint_path_for_cycle = current_checkpoint_path # Fallback
                 cycle_step_failed = True
        else:
             # If prior steps failed, just use the checkpoint we started with
             final_checkpoint_path_for_cycle = current_checkpoint_path
             logging.warning(f"Skipping checkpoint selection/finalization due to previous failure in cycle.")


        # --- Step 6: Test the Trained Model ---
        logging.info(f"[Cycle {cycle + 1}] Step 6: Testing model using {final_checkpoint_path_for_cycle}...")
        # Only test if checkpoint selection didn't fail critically AND the path is valid
        if not cycle_step_failed and final_checkpoint_path_for_cycle and final_checkpoint_path_for_cycle.exists():
            cmd_test = [
                'python', test_script,
                '--checkpoint', str(final_checkpoint_path_for_cycle), # Use finalized path (e.g., best.ckpt)
                '--data_root', str(cycle_test_dir),
                '--test_folder', test_lmdb_foldername,
                '--batch_size', str(config.get('test_batch_size', 64)),
                '--num_workers', str(config.get('num_workers', 8)),
                '--device', str(config.get('device', 'cuda'))
            ]
            logging.info(f"Running test command...")
            if not run_command(cmd_test):
                 logging.error(f"[Cycle {cycle + 1}] Testing failed (command execution error).")
                 # Note: Testing failure doesn't necessarily stop the next cycle unless checkpoint is invalid
        elif cycle_step_failed:
             logging.warning(f"[Cycle {cycle + 1}] Skipping testing because a previous step in the cycle failed.")
        else:
            logging.warning(f"[Cycle {cycle + 1}] Skipping testing as no valid checkpoint was finalized ({final_checkpoint_path_for_cycle}).")

        # --- Prepare for Next Cycle ---
        # Update the path for the *next* cycle's input checkpoint
        if not cycle_step_failed and final_checkpoint_path_for_cycle and final_checkpoint_path_for_cycle.exists():
             current_checkpoint_path = final_checkpoint_path_for_cycle # Use the finalized path (best.ckpt or fallback)
             logging.info(f"Checkpoint for next cycle set to: {current_checkpoint_path}")
        else:
             # Keep the old checkpoint path if this cycle had issues finding/finalizing a new one
             logging.warning(f"Could not find or finalize a new checkpoint from this cycle. Next cycle will reuse: {current_checkpoint_path}")

        # Update the variable tracking the data root for cleanup in the *next* iteration
        prev_cycle_data_root = cycle_data_root

        logging.info(f"--- Finished Cycle {cycle + 1}/{num_cycles} ---")
        # --- End of Cycle Loop ---

    # --- Final Cleanup (Optional) ---
    if prev_cycle_data_root and config.get('cleanup_intermediate', True):
        if prev_cycle_data_root.exists():
            logging.info(f"Cleaning up last cycle's combined data directory: {prev_cycle_data_root}")
            try: shutil.rmtree(prev_cycle_data_root)
            except Exception as e: logging.error(f"Failed to remove last cycle data {prev_cycle_data_root}: {e}")
        else: logging.warning(f"Last cycle data directory not found for cleanup: {prev_cycle_data_root}")

    logging.info("Pseudo-Labeling Pipeline Finished.")


if __name__ == "__main__":
    main()