import torch
ckpt_path = "/ssd_scratch/cvit/lalitha/2_pdf_scraping_data/hindi_ckpt/parseq.ckpt"
checkpoint = torch.load(ckpt_path, map_location="cpu")
print(checkpoint.keys())  # Should print model keys if it's a valid checkpoint
