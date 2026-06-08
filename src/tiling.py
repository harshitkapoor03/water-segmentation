# import os
# import numpy as np
# from PIL import Image
# from tqdm import tqdm
# import logging

# logger = logging.getLogger(__name__)


# def get_patch_coords(image_h, image_w, patch_size, stride):
#     """
#     Returns a list of (y, x) top-left coordinates for every patch
#     that should be extracted from an image of size (image_h, image_w).

#     This is the core tiling logic. We separate coordinate generation
#     from the actual patch extraction so the same function can be
#     reused for both training (extract and save) and inference
#     (extract, predict, stitch).

#     How it works:
#     Step 1 — Normal grid: slide the window across with the given stride.
#              Loop stops when adding one more stride would go out of bounds.

#     Step 2 — Right edge: after the normal horizontal loop, check if the
#              last patch reached the right edge of the image.
#              If image_w - last_x > 0, there is an uncovered strip.
#              Add one more patch anchored to the right edge:
#              x = image_w - patch_size
#              This patch overlaps more with the previous one but that's fine.

#     Step 3 — Bottom edge: same logic applied vertically.

#     Step 4 — Corner: if both a right-edge gap AND a bottom-edge gap exist,
#              the bottom-right corner needs a patch too.
#              x = image_w - patch_size, y = image_h - patch_size.

#     Why we use a set() to collect coordinates:
#     It's possible that an edge patch lands on exactly the same position
#     as a normal grid patch (when dimensions divide evenly). Using a set
#     prevents duplicates — a patch at the same position twice would be
#     counted twice in the accumulator, inflating its weight.

#     set() stores tuples. Tuples are hashable so they work as set elements.
#     After collecting all unique coords, we convert back to a sorted list
#     for consistent ordering.
#     """
#     coords = set()

#     # --- Step 1: Normal sliding window grid ---
#     # range(start, stop, step) — generates values from start up to but
#     # not including stop. We stop when y + patch_size would exceed image_h,
#     # i.e., when y > image_h - patch_size.
#     # Equivalent stop value: image_h - patch_size + 1
#     for y in range(0, image_h - patch_size + 1, stride):
#         for x in range(0, image_w - patch_size + 1, stride):
#             coords.add((y, x))

#     # --- Step 2: Right edge patches ---
#     # After the x loop, what was the last x value?
#     # last_x = the largest multiple of stride that fits: floor((W-P)/stride)*stride
#     # Check if image_w - (last_x + patch_size) > 0 meaning there's a gap.
#     # Simpler equivalent: just always add x = image_w - patch_size for every y
#     # row, then the set deduplication handles the case where it already exists.

#     # We iterate over all y values that were used in the normal grid
#     # (same y range as Step 1) plus the bottom edge y if needed
#     all_y_values = list(range(0, image_h - patch_size + 1, stride))

#     # right edge x — anchored to image right boundary
#     right_x = image_w - patch_size

#     for y in all_y_values:
#         coords.add((y, right_x))

#     # --- Step 3: Bottom edge patches ---
#     bottom_y = image_h - patch_size

#     all_x_values = list(range(0, image_w - patch_size + 1, stride))

#     for x in all_x_values:
#         coords.add((bottom_y, x))

#     # --- Step 4: Bottom-right corner patch ---
#     coords.add((bottom_y, right_x))

#     # Sort for consistent ordering — sorted() on tuples sorts by first element
#     # then second element, so this gives us top-to-bottom, left-to-right order
#     return sorted(coords)


# def extract_patches_from_coords(image_array, mask_array, coords, patch_size):
#     """
#     Given a list of (y, x) coordinates, extracts the corresponding
#     patches from image_array and mask_array.

#     image_array: numpy array shape (H, W, 3)
#     mask_array:  numpy array shape (H, W)
#     coords:      list of (y, x) top-left corner positions
#     patch_size:  integer, side length of each square patch

#     Returns list of (image_patch, mask_patch) tuples.
#     """
#     patches = []
#     for y, x in coords:
#         img_patch = image_array[y : y + patch_size, x : x + patch_size]
#         mask_patch = mask_array[y : y + patch_size, x : x + patch_size]
#         patches.append((img_patch, mask_patch))
#     return patches


# def process_single_image(image_array, mask_array, patch_size, stride):
#     """
#     Decides whether to tile or resize a single image+mask pair.

#     If both dimensions >= patch_size:
#         Generate patch coordinates with edge coverage, extract patches.

#     If either dimension < patch_size:
#         Image is smaller than one patch — can't tile it.
#         Resize up to patch_size as the only option.
#         We use BILINEAR for the image (smooth interpolation)
#         and NEAREST for the mask (avoids creating values between 0 and 255
#         that would corrupt the binary water/land labels).

#     Returns list of (image_patch, mask_patch) tuples.
#     """
#     h, w = image_array.shape[:2]

#     if h >= patch_size and w >= patch_size:
#         coords = get_patch_coords(h, w, patch_size, stride)
#         return extract_patches_from_coords(image_array, mask_array, coords, patch_size)
#     else:
#         resized_img = np.array(
#             Image.fromarray(image_array).resize(
#                 (patch_size, patch_size), Image.BILINEAR
#             )
#         )
#         resized_mask = np.array(
#             Image.fromarray(mask_array).resize((patch_size, patch_size), Image.NEAREST)
#         )
#         return [(resized_img, resized_mask)]


# def build_patches(config):
#     """
#     Runs the full tiling pipeline over the entire raw dataset.
#     Saves each patch as a PNG to the patches directories.
#     Called once from train.py before training starts.

#     If patches already exist, train.py skips this function.
#     """
#     raw_img_dir = config["data"]["raw_image_dir"]
#     raw_mask_dir = config["data"]["raw_mask_dir"]
#     out_img_dir = config["data"]["patches_image_dir"]
#     out_mask_dir = config["data"]["patches_mask_dir"]
#     patch_size = config["data"]["patch_size"]
#     stride = config["data"]["stride"]

#     os.makedirs(out_img_dir, exist_ok=True)
#     os.makedirs(out_mask_dir, exist_ok=True)

#     image_files = sorted(os.listdir(raw_img_dir))
#     total_patches = 0

#     for fname in tqdm(image_files, desc="Building patches"):
#         img_path = os.path.join(raw_img_dir, fname)
#         mask_path = os.path.join(raw_mask_dir, fname)

#         if not os.path.exists(mask_path):
#             logger.warning(f"No mask for {fname}, skipping")
#             continue

#         image_array = np.array(Image.open(img_path).convert("RGB"))
#         mask_array = np.array(Image.open(mask_path).convert("L"))

#         patches = process_single_image(image_array, mask_array, patch_size, stride)

#         base = os.path.splitext(fname)[0]
#         for i, (img_p, mask_p) in enumerate(patches):
#             patch_name = f"{base}_p{i:04d}.png"
#             Image.fromarray(img_p).save(os.path.join(out_img_dir, patch_name))
#             Image.fromarray(mask_p).save(os.path.join(out_mask_dir, patch_name))
#             total_patches += 1

#     logger.info(f"Done. {total_patches} patches from {len(image_files)} images")
#     return total_patches


# def tile_for_inference(image_array, patch_size, stride):
#     """
#     Tiles one image for inference.
#     Returns patches AND their coordinates so we can stitch later.

#     Same coordinate logic as training — get_patch_coords handles
#     edge coverage identically. This consistency matters:
#     the model was trained on patches generated this way,
#     so inference should generate patches the same way.
#     """
#     h, w = image_array.shape[:2]

#     if h >= patch_size and w >= patch_size:
#         coords = get_patch_coords(h, w, patch_size, stride)
#         patches = [
#             image_array[y : y + patch_size, x : x + patch_size] for (y, x) in coords
#         ]
#     else:
#         # Image smaller than patch — resize and treat as single patch
#         resized = np.array(
#             Image.fromarray(image_array).resize(
#                 (patch_size, patch_size), Image.BILINEAR
#             )
#         )
#         patches = [resized]
#         coords = [(0, 0)]

#     return patches, coords


# def stitch_predictions(patch_probs, coords, original_h, original_w, patch_size):
#     """
#     Reconstructs a full-resolution probability map from predicted patch probs.

#     Why we average probabilities instead of stitching binary masks:
#     If we threshold each patch separately and stitch binary 0/1 masks,
#     we get visible seams — patch A might predict pixel X as water,
#     patch B might predict the same pixel as land, and the boundary
#     between patches becomes a hard visible line in the output.

#     Instead:
#     1. Accumulate probability values across all patches that cover each pixel
#     2. Divide by how many patches covered that pixel (the count)
#     3. Threshold ONCE on the averaged probability map

#     In overlap zones (including the extra-overlap edge patches you added),
#     we get 2, 4, or more probability estimates for the same pixel and
#     average them. More estimates → more stable prediction.
#     Edge pixels that were only covered by the edge patch get one estimate.
#     Either way the threshold is applied consistently to the averaged map.

#     accumulator dtype float32:
#     Probabilities are 0.0 to 1.0. Summing many float32 values is safe.
#     We can't use uint8 here — it would overflow immediately.
#     """
#     accumulator = np.zeros((original_h, original_w), dtype=np.float32)
#     count = np.zeros((original_h, original_w), dtype=np.float32)

#     for prob_map, (y, x) in zip(patch_probs, coords):
#         accumulator[y : y + patch_size, x : x + patch_size] += prob_map
#         count[y : y + patch_size, x : x + patch_size] += 1.0

#     # np.maximum(count, 1.0) prevents division by zero for any pixel
#     # that somehow wasn't covered — sets those to count=1 so they
#     # stay at 0/1 = 0 probability (predicted as land)
#     avg_prob = accumulator / np.maximum(count, 1.0)

#     return avg_prob
