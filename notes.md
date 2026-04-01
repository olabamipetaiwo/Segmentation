 What was created                                                                                                                                                                            
                                                                                                                                                                                              
  10 Python files, all syntax-clean:                                                                                                                                                          
                                                                                                                                                                                              
  ┌─────────────────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────┐                                                              
  │              File               │                                         Purpose                                          │                                                              
  ├─────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤                                                              
  │ preprocessing/generate_masks.py │ Parse NuInsSeg XML annotations, generate binary/instance/distance masks                  │                                                              
  ├─────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤                                                              
  │ datasets/nuinsseg_dataset.py    │ PyTorch Dataset with albumentations augmentation, SAM normalisation, centroid extraction │                                                              
  ├─────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤                                                              
  │ models/lora.py                  │ LoRALayer, QKVLoRAWrapper, inject_lora                                                   │                                                              
  ├─────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤                                                              
  │ models/mobile_sam_lora.py       │ MobileSAM loaded with LoRA + unfrozen mask decoder                                       │                                                              
  ├─────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤                                                              
  │ utils/losses.py                 │ DiceLoss, FocalLoss, CombinedLoss                                                        │                                                              
  ├─────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤                                                              
  │ utils/metrics.py                │ dice_coefficient, aggregated_jaccard_index, panoptic_quality, compute_all_metrics        │
  ├─────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤                                                              
  │ utils/prompt_utils.py           │ Centroid extraction, prompt tensor building, grid generation                             │
  ├─────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤                                                              
  │ train.py                        │ 5-fold CV loop, AdamW + cosine LR, CSV logging, best/periodic checkpoints                │
  ├─────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤                                                              
  │ evaluate.py                     │ Grid prompts + IoU filtering + mask NMS, per-image CSV output                            │
  ├─────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤                                                              
  │ visualize.py                    │ 3-panel comparison figures, selects strong and weak examples                             │
  └─────────────────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────┘                                                              
                                                            
  Key LoRA design (worth highlighting in your report): MobileSAM's TinyViT uses a fused qkv = Linear(dim, 3*dim). The injection splits this into Q (LoRA), K (frozen), V (LoRA) and replaces  
  it with a QKVLoRAWrapper that concatenates outputs to preserve the original interface. Only ~4M parameters are trained (LoRA + mask decoder) out of ~5.7M total.
                                                                                                                                                                                              
  Next steps to run it                                                                                                                                                                        
  
  1. Download NuInsSeg to data/nuinsseg/                                                                                                                                                      
  2. Clone MobileSAM and install: pip install -e .; place mobile_sam.pt in project root
  3. pip install -r requirements.txt                                                                                                                                                          
  4. python preprocessing/generate_masks.py                                                                                                                                                   
  5. python train.py --checkpoint mobile_sam.pt                                                                                                                                               
  6. python evaluate.py --checkpoint checkpoints/fold_1/best_checkpoint.pt --fold_index 0                                                                                                     
  7. python visualize.py --checkpoint checkpoints/fold_1/best_checkpoint.pt --fold_index 0  


  All 6 shell scripts are in shell/ and pass syntax checks. Here's a summary:                                                                                                                 
                                                                                                                                                                                              
  ┌───────────────────────┬─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐                                                 
  │        Script         │                                                  What it does                                                   │                                                 
  ├───────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤                                                 
  │ shell/setup.sh        │ Creates venv, detects CUDA, installs PyTorch + MobileSAM + requirements.txt, reminds you to place mobile_sam.pt │                                                 
  ├───────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤                                                 
  │ shell/preprocess.sh   │ Runs preprocessing/generate_masks.py to generate masks from XML                                                 │                                                 
  ├───────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤                                                 
  │ shell/train.sh        │ Runs train.py; supports --fold N for a single fold                                                              │                                                 
  ├───────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤                                                 
  │ shell/evaluate.sh     │ Runs evaluate.py for one fold; supports all inference flags                                                     │                                                 
  ├───────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤                                                 
  │ shell/visualize.sh    │ Runs visualize.py for a fold                                                                                    │
  ├───────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤                                                 
  │ shell/run_pipeline.sh │ Chains all steps end-to-end; supports --skip_setup and --skip_preprocess                                        │
  └───────────────────────┴─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘                                                 
                                                            
  Typical workflow:                                                                                                                                                                           
                                                            
  # First time only
  bash shell/setup.sh                                                                                                                                                                         
                                                                                                                                                                                              
  # Download data to data/nuinsseg/, then:                                                                                                                                                    
  bash shell/preprocess.sh                                                                                                                                                                    
                                                                                                                                                                                              
  # Train all folds
  bash shell/train.sh --checkpoint mobile_sam.pt                                                                                                                                              
                                                            
  # Evaluate fold 1
  bash shell/evaluate.sh --checkpoint checkpoints/fold_1/best_checkpoint.pt --fold_index 0
                                                                                                                                                                                              
  # Generate figures
  bash shell/visualize.sh --checkpoint checkpoints/fold_1/best_checkpoint.pt --fold_index 0                                                                                                   
                                                                                                                                                                                              
  # Or run everything at once (skip steps already done)                                                                                                                                       
  bash shell/run_pipeline.sh --skip_setup --skip_preprocess    