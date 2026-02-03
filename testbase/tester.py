from datetime import datetime
import sys
import os
import cv2
from pathlib import Path
from engine import Engine

# Add codebase to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'codebase'))


def get_now()->str:
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def test_im(path: str, modeleattr: list[str], save=False):
    """
    Test Engine image processing on a single image.
    
    Args:
        path: Path to input image file
        dst: Module name to test (e.g., "fixation", "skew", "preprocess", etc.)
        save: If True, save results to test folder
    """
    try:
        # Read image
        if not os.path.exists(path):
            print(f"[ERROR] Image not found: {path}")
            return
        
        img = cv2.imread(path)
        if img is None:
            print(f"[ERROR] Could not read image: {path}")
            return
        
        print(f"[TEST] Testing module: {modeleattr}")
        print(f"[TEST] Input image: {path} | Size: {img.shape}")
        
        # Create engine instance
        engine = Engine()
        

        
        for m in modeleattr:
            # Call the appropriate method
            if not hasattr(engine, m):
                print(f"[ERROR] Module '{m}' not found in Engine class")
                return
            method = getattr(engine, m)
            img = method(img)
            
        # Handle different return types
        if isinstance(img, tuple):
            processed_img = img[0]
        else:
            processed_img = img
        
        # Ensure output is 3-channel
        if len(processed_img.shape) == 2:
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
        elif processed_img.shape[2] != 3:
            print(f"[WARNING] Output channels: {processed_img.shape[2]}, converting to 3 channels")
            if processed_img.shape[2] == 1:
                processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
        
        print(f"[SUCCESS] Module '{modeleattr}' processed successfully")
        print(f"[OUTPUT] Result shape: {processed_img.shape}")
        
        # Save if requested
        if save:
            timestamp = get_now()
            save_folder = os.path.join(os.path.dirname(__file__), f"results_{timestamp}")
            os.makedirs(save_folder, exist_ok=True)
            
            output_path = os.path.join(save_folder, f"{modeleattr}_output.jpg")
            cv2.imwrite(output_path, processed_img)
            print(f"[SAVED] Result saved to: {output_path}")
        
        return processed_img
        
    except Exception as e:
        print(f"[ERROR] Test failed for module '{modeleattr}': {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_batch(img_folder: str, dst_folder: str,moduleatts:list[str], save=False):
    """
    Test Engine image processing on all images in a folder.
    
    Args:
        img_folder: Path to folder containing images
        dst_folder: Module name to test
        save: If True, save results to test folder
    """
    try:
        if not os.path.exists(img_folder):
            print(f"[ERROR] Image folder not found: {img_folder}")
            return
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [
            f for f in os.listdir(img_folder) 
            if os.path.splitext(f)[1].lower() in image_extensions
        ]
        
        if not image_files:
            print(f"[ERROR] No images found in: {img_folder}")
            return
        
        print(f"[TEST] Testing module: {dst_folder}")
        print(f"[TEST] Found {len(image_files)} images to process")
        
        # Create batch results folder if saving
        results_folder = None
        if save:
            timestamp = get_now()
            results_folder = os.path.join(os.path.dirname(__file__), f"batch_results_{dst_folder}_{timestamp}")
            os.makedirs(results_folder, exist_ok=True)
            print(f"[INFO] Results will be saved to: {results_folder}")
        
        # Process each image
        engine = Engine()
        success_count = 0
        failed_count = 0
        
        for idx, img_file in enumerate(image_files, 1):
            img_path = os.path.join(img_folder, img_file)
            print(f"\n[BATCH {idx}/{len(image_files)}] Processing: {img_file}")
            
            try:
                # Call the appropriate method
                if not hasattr(engine, dst_folder):
                    print(f"  [ERROR] Module '{dst_folder}' not found in Engine class")
                    failed_count += 1
                    continue
                
                result = test_im(img_path,moduleatts,save)
                # method = getattr(engine, dst_folder)
                # result = method(img)
                
                # Handle different return types
                if isinstance(result, tuple):
                    processed_img = result[0]
                else:
                    processed_img = result
                
                # Ensure output is 3-channel
                if len(processed_img.shape) == 2:
                    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
                elif processed_img.shape[2] != 3:
                    if processed_img.shape[2] == 1:
                        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
                
                print(f"  [SUCCESS] Processed | Output shape: {processed_img.shape}")
                
                # Save if requested
                if save:
                    base_name = os.path.splitext(img_file)[0]
                    output_path = os.path.join(results_folder, f"{base_name}_{dst_folder}_output.jpg")
                    cv2.imwrite(output_path, processed_img)
                    print(f"  [SAVED] {output_path}")
                
                success_count += 1
                
            except Exception as e:
                print(f"  [FAILED] {str(e)}")
                failed_count += 1
        
        # Summary
        print(f"\n[SUMMARY] Batch processing complete for module '{dst_folder}':")
        print(f"  Success: {success_count}/{len(image_files)}")
        print(f"  Failed: {failed_count}/{len(image_files)}")
        if save and results_folder:
            print(f"  Results folder: {results_folder}")
        
    except Exception as e:
        print(f"[ERROR] Batch test failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
