"""
dataset.py
----------
Loads and preprocesses the Retina Age Analysis dataset.

The HuggingFace repo (ramankamran/retina-age-analysis) is laid out like this:
    images/            9,857 JPEGs, named img00001.jpg, img00002.jpg, ...
                       (one flat folder - NOT already split into train/val/test)
    splits/train.csv   image_id, patient_id, patient_age, + clinical columns
    splits/val.csv     same columns, validation patients
    splits/test.csv    same columns, test patients

There is no ready-made "image" column and no dataset loading script in the
repo, so calling datasets.load_dataset(...) on its own only returns the
metadata columns, without any images. Instead we download the images/ and
splits/ folders with huggingface_hub, read the 3 split CSVs with pandas,
and load each image directly from images/<image_id>.jpg with PIL. This
matches the loading example shown on the dataset's own README
(pd.read_csv("hf://datasets/.../splits/train.csv")).
"""

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

HF_REPO_ID = "ramankamran/retina-age-analysis"


def _on_kaggle():
    return Path("/kaggle/working").is_dir()


def writable_root():
    """
    Where we are allowed to write large files.

    On Kaggle /kaggle/input is read-only (that is where attached datasets
    appear), and /kaggle/working is the writable directory that also gets
    saved as the notebook's output - so it is both the scratch space and
    the way results survive the session.
    """
    return Path("/kaggle/working") if _on_kaggle() else Path.home() / ".cache"


DEFAULT_CACHE_DIR = writable_root() / "retina_age_dataset"


def find_prepared_dataset():
    """
    Looks for an already-prepared copy of the data attached as a Kaggle
    dataset - any folder under /kaggle/input that contains splits/.

    This is the whole point of moving to Kaggle: we download and resize
    once, save the result as a dataset, and then every later notebook
    attaches it instead of pulling ~10GB from HuggingFace again.
    """
    base = Path("/kaggle/input")
    if not base.is_dir():
        return None

    # The mount depth is not fixed and is deeper than it looks. Kaggle mounts
    # a dataset at /kaggle/input/datasets/<user>/<name>/... and a notebook's
    # output deeper still, so hard-coding one or two levels does not work.
    # Search a bounded range of depths instead of rglob: a fixed-depth glob
    # only inspects directory names at that level, while rglob would walk
    # into images_224/ and stat ~10,000 files on network storage.
    patterns = ["splits"] + ["/".join(["*"] * n) + "/splits" for n in range(1, 7)]
    candidates = [p.parent for pat in patterns
                  for p in sorted(base.glob(pat)) if p.is_dir()]
    if not candidates:
        return None

    # Prefer one that also ships a prebuilt resized cache - that is the copy
    # that needs no work at all.
    for c in candidates:
        if any(c.glob("images_*")):
            return c
    return candidates[0]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


EXPECTED_IMAGE_COUNT = 9800  # a bit under the real 9,857, as a safety margin


def _looks_fully_downloaded(cache_dir):
    """Cheap local check so we don't ask HuggingFace to re-verify all 9,857
    files (over the network) every time get_dataloaders() is called."""
    images_dir = cache_dir / "images"
    splits_dir = cache_dir / "splits"
    if not images_dir.is_dir() or not splits_dir.is_dir():
        return False
    if not all((splits_dir / f"{s}.csv").exists() for s in ("train", "val", "test")):
        return False
    return sum(1 for _ in images_dir.glob("*.jpg")) >= EXPECTED_IMAGE_COUNT


def download_dataset(cache_dir=None):
    """
    Downloads the images/ and splits/ folders to a local directory.

    If everything is already on disk (checked locally, no network call),
    this returns immediately. Otherwise it calls huggingface_hub, which
    downloads (or resumes) images/ and splits/.

    This local check matters: every training run calls get_dataloaders(),
    which calls this function, so without it a full run (baseline + 3 CNN
    configs + 3 ablation experiments = ~10 calls) would ask HuggingFace to
    re-verify all 9,865 files 10 times over, which is slow and can trigger
    "429 rate limited" errors.
    """
    # An attached, already-prepared dataset wins over everything: nothing
    # to download and nothing to resize.
    prepared = find_prepared_dataset()
    if prepared is not None:
        return prepared

    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR

    if _looks_fully_downloaded(cache_dir):
        return cache_dir

    from huggingface_hub import snapshot_download

    root = snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=str(cache_dir),
        allow_patterns=["images/*", "splits/*"],
    )
    return Path(root)


def load_split_dataframe(split, root_dir):
    """split is one of 'train', 'val', 'test' (matches the CSV file names)."""
    csv_path = Path(root_dir) / "splits" / f"{split}.csv"
    return pd.read_csv(csv_path)


def _resize_one(args):
    """Resize a single image into the cache. Top-level (not a closure) so
    it can be handed to a thread pool."""
    src_path, dst_path, img_size = args
    try:
        img = Image.open(src_path)
        # draft() lets libjpeg decode straight to a smaller size instead of
        # unpacking the full ~1900x1900 image and shrinking it afterwards.
        # Measured ~2.6x faster on images this size, and since we resize to
        # img_size right after, the intermediate size doesn't matter.
        img.draft("RGB", (img_size, img_size))
        img = img.convert("RGB").resize((img_size, img_size))
        # Write to a temp name and rename: renaming is atomic, so stopping
        # the cell mid-build can never leave a half-written JPEG behind
        # that later looks "already done" and crashes during training.
        tmp_path = dst_path.with_suffix(".tmp")
        # Pass the format explicitly - PIL normally infers it from the file
        # extension, and ".tmp" means nothing to it.
        img.save(tmp_path, "JPEG", quality=90)
        tmp_path.replace(dst_path)
        return True
    except Exception as e:
        # A single unreadable file shouldn't kill a cache build that takes
        # minutes - report it and let the rest finish.
        print(f"  skipped {src_path.name}: {e}")
        return False


def resolve_images_dir(root, img_size):
    """
    Returns the folder the DataLoaders should read images from.

    Order of preference:
      1. root/images_<size>  - already resized (an attached dataset, or a
         cache we built in an earlier session). Nothing to do.
      2. build the cache from root/images into a writable location. On
         Kaggle `root` may be read-only, so the cache cannot simply live
         next to the originals.
    """
    root = Path(root)
    prepared = root / f"images_{img_size}"
    if prepared.is_dir() and (prepared / ".complete").exists():
        return prepared

    originals = root / "images"
    if not originals.is_dir():
        raise FileNotFoundError(
            f"no images_{img_size}/ and no images/ under {root}. If this is an "
            f"attached dataset, it needs one of them."
        )

    # /kaggle/input is always read-only, whatever a permission probe says
    # (it reports writable when the kernel happens to run as root). Anything
    # under it gets its cache written to the working directory instead.
    under_kaggle_input = str(root).startswith("/kaggle/input")
    writable = (not under_kaggle_input) and os.access(root, os.W_OK)

    cache_root = root if writable else DEFAULT_CACHE_DIR
    cache_root.mkdir(parents=True, exist_ok=True)
    return _ensure_resized_cache(originals, img_size, cache_root=cache_root)


def _ensure_resized_cache(images_dir, img_size, cache_root=None):
    """
    Resizes every original image once to img_size x img_size and saves it
    to a sibling "images_<size>" folder, then returns that folder.

    The original photos are ~1900x1900 - decoding a JPEG that big is the
    slow part of loading a batch, not the actual resize, so re-decoding
    the full-size file on every training epoch wastes a huge amount of
    time redoing identical work. Measured on Colab: ~2.3s per batch of 32
    from the originals vs ~0.3s from the cached copies, i.e. roughly 10
    minutes per epoch down to well under one. We only noticed this was the
    bottleneck after seeing nvidia-smi report 0% GPU utilization while
    "training" - the GPU was idle, waiting on JPEG decoding.

    Already-resized images are skipped, so re-running this after a Colab
    disconnect, or after it was interrupted partway, picks up where it
    left off instead of starting over.
    """
    cache_root = Path(cache_root) if cache_root else images_dir.parent
    cache_dir = cache_root / f"images_{img_size}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    done_flag = cache_dir / ".complete"

    # Once the cache is built, skip the directory scan entirely. Without
    # this, every experiment (~9 of them) would stat all 9,857 files again
    # just to learn there's nothing to do.
    if done_flag.exists():
        return cache_dir

    all_jpgs = sorted(images_dir.glob("*.jpg"))
    todo = [(p, cache_dir / p.name, img_size)
            for p in all_jpgs if not (cache_dir / p.name).exists()]

    if todo:
        from concurrent.futures import ThreadPoolExecutor
        from tqdm import tqdm

        # Threads (not processes): PIL releases the GIL while decoding, so
        # this scales with the CPU count, and unlike a process pool it
        # can't leave orphaned workers hanging around in a notebook kernel.
        n_threads = max(2, (os.cpu_count() or 2))
        print(f"Building a {img_size}x{img_size} image cache for {len(todo)} images "
              f"({n_threads} threads) - this only happens once.")
        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            results = list(tqdm(pool.map(_resize_one, todo),
                                 total=len(todo), desc="pre-resizing"))

        n_failed = results.count(False)
        if n_failed:
            # Don't mark the cache complete if anything failed - otherwise
            # the next run trusts it and training crashes on a missing file.
            raise RuntimeError(
                f"{n_failed}/{len(todo)} images failed to resize - cache not marked "
                f"complete. Re-run this cell to retry just the missing ones."
            )

    done_flag.touch()
    return cache_dir


def get_transforms(train, img_size=224):
    """
    train=False: resize + normalize only (used for val/test, and for the
                 no-augmentation configs).
    train=True: adds the light augmentation described in the proposal -
                small rotation, horizontal flip, brightness jitter.
    """
    steps = [transforms.Resize((img_size, img_size))]
    if train:
        steps += [
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2),
        ]
    steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(steps)


class RetinaAgeDataset(Dataset):
    """Wraps one split's dataframe + the local images/ folder, returns (image_tensor, age)."""

    def __init__(self, df, images_dir, transform, return_features=False):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.return_features = return_features

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.images_dir / f"{row['image_id']}.jpg"
        img = Image.open(img_path).convert("RGB")
        age = float(row["patient_age"])

        if self.return_features:
            return extract_basic_features(img), torch.tensor(age, dtype=torch.float32)

        return self.transform(img), torch.tensor(age, dtype=torch.float32)


def extract_basic_features(img, thumb_size=16, stats_size=128):
    """
    Hand-crafted features for the non-deep linear regression baseline
    (proposal: "linear regression on basic image features"):
      - a flattened grayscale thumbnail
      - per-channel mean/std
      - overall brightness and contrast

    The color/brightness stats are computed on a resized (stats_size x
    stats_size) copy rather than the original image. The original photos
    are a few thousand pixels wide, so doing this on the full image made
    feature extraction very slow across ~8,400 images for barely any
    difference in the resulting statistics.
    """
    thumb = np.asarray(img.resize((thumb_size, thumb_size))).astype(np.float32) / 255.0
    gray_flat = thumb.mean(axis=2).flatten()

    full = np.asarray(img.resize((stats_size, stats_size))).astype(np.float32) / 255.0
    channel_means = full.reshape(-1, 3).mean(axis=0)
    channel_stds = full.reshape(-1, 3).std(axis=0)
    brightness = full.mean()
    contrast = full.std()

    feats = np.concatenate([gray_flat, channel_means, channel_stds, [brightness, contrast]])
    return torch.tensor(feats, dtype=torch.float32)


def _seed_worker(worker_id):
    """
    Seeds each DataLoader worker.

    Workers are separate processes with their own RNG state. Without this,
    the random augmentations (rotation, flip, brightness) differ from run to
    run even when the main process seed is fixed - so cnn_aug would not be
    reproducible and we would not notice, because every other config is
    deterministic and looks fine.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloaders(batch_size=32, img_size=224, augment_train=False,
                     num_workers=None, return_features=False, cache_dir=None,
                     seed=0):
    """
    Downloads the dataset (if not already cached) and builds train/val/test
    DataLoaders from the 3 split CSVs - these already assign patients to a
    single split, so there's no risk of one patient's images leaking across
    train/val/test.
    """
    root = download_dataset(cache_dir=cache_dir)
    images_dir = root / "images"

    # Everything reads from the same resized copies - including the linear
    # baseline. It builds its own small thumbnails so it doesn't need the
    # cache for speed, but sharing it matters for two reasons.
    #
    # Fairness: the non-deep baseline should not get higher-resolution input
    # than the CNNs it is compared against. If it read the ~1900x1900
    # originals while the networks saw 224x224, any gap between them would
    # partly be a difference in available detail rather than in method.
    #
    # Practicality: the prepared dataset ships only images_224, so there are
    # no originals to read. Requiring them meant the baseline crashed on a
    # missing file while every CNN config ran fine.
    images_dir = resolve_images_dir(root, img_size)

    train_df = load_split_dataframe("train", root)
    val_df = load_split_dataframe("val", root)
    test_df = load_split_dataframe("test", root)

    train_tf = get_transforms(train=augment_train, img_size=img_size)
    eval_tf = get_transforms(train=False, img_size=img_size)

    train_ds = RetinaAgeDataset(train_df, images_dir, train_tf, return_features)
    val_ds = RetinaAgeDataset(val_df, images_dir, eval_tf, return_features)
    test_ds = RetinaAgeDataset(test_df, images_dir, eval_tf, return_features)

    # Default to one worker per CPU (Colab's free tier gives 2). Asking for
    # more than that doesn't help - the workers just compete for the same
    # cores - which is why bumping this to 4 by hand made no difference.
    if num_workers is None:
        num_workers = max(2, os.cpu_count() or 2)

    # pin_memory speeds up the CPU->GPU transfer, and persistent_workers
    # avoids re-spawning the worker processes at the start of every epoch -
    # both cheap wins once we noticed the GPU sitting idle during training.
    use_persistent = num_workers > 0

    # An explicit generator for the shuffle, rather than the global RNG.
    # The global stream has already been consumed by building the model, and
    # different architectures consume different amounts of it - so without
    # this the training data arrives in a different order for each config,
    # and part of any gap between two configs would be the data order rather
    # than the factor we changed.
    shuffle_gen = torch.Generator()
    shuffle_gen.manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True,
                               persistent_workers=use_persistent,
                               generator=shuffle_gen, worker_init_fn=_seed_worker)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True,
                             persistent_workers=use_persistent)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=use_persistent)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Sanity check - needs internet access, so run this in Colab.
    print(f"Downloading {HF_REPO_ID} (images/ + splits/) ...")
    root = download_dataset()
    train_df = load_split_dataframe("train", root)
    print("train.csv columns:", list(train_df.columns))
    print("train rows:", len(train_df))

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=4, augment_train=True)
    imgs, ages = next(iter(train_loader))
    print("Batch image shape:", imgs.shape)
    print("Batch ages:", ages)
