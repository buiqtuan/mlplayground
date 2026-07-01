# FACE DETECTION WITH PYTORCH — STEP-BY-STEP GUIDE
#
# Goal: given an input image, detect whether/where a human face appears in it.
# This file has no code on purpose — follow the steps below and write the
# code yourself underneath each comment. Work top to bottom, run often.


# ---------------------------------------------------------------------------
# STEP 0: Decide what "detect a face" means for your first version
# ---------------------------------------------------------------------------
# There are two different problems people call "face detection":
#   1. Classification: "does this image contain a face?" (yes/no)
#   2. Detection: "where is the face?" (a bounding box, maybe multiple faces)
# Start with (1) — it's a plain image classifier and the fastest way to get
# something working end to end. You can upgrade to (2) afterwards (see the
# "Going further" section at the bottom).


# ---------------------------------------------------------------------------
# STEP 1: Set up your environment
# ---------------------------------------------------------------------------
# - Create/activate a virtual environment.
# - Install: torch, torchvision, pillow, matplotlib (for viewing images),
#   and optionally opencv-python (for webcam/video and easy image loading).
# - Verify torch can see a GPU if you have one: check torch.cuda.is_available().


# ---------------------------------------------------------------------------
# STEP 2: Get a dataset of faces vs. non-faces
# ---------------------------------------------------------------------------
# You need two classes of images:
#   - "face" images: e.g. a subset of the CelebA dataset, or LFW
#     (Labeled Faces in the Wild), both loadable via torchvision.datasets.
#   - "no face" images: any generic image dataset without people, e.g. a
#     handful of classes from CIFAR-10 (cars, animals, objects).
# For a first pass, keep it small — a few thousand images per class is
# plenty to prove the pipeline works before scaling up.
# Organize images into folders like:
#   data/train/face/...
#   data/train/no_face/...
#   data/val/face/...
#   data/val/no_face/...
# so you can use torchvision.datasets.ImageFolder later.


# ---------------------------------------------------------------------------
# STEP 3: Build the data loading pipeline
# ---------------------------------------------------------------------------
# - Define image transforms: resize to a fixed size (e.g. 128x128), convert
#   to tensor, normalize pixel values.
# - Use torchvision.datasets.ImageFolder to load train/ and val/ folders.
# - Wrap each in a torch.utils.data.DataLoader with a batch size and
#   shuffle=True for training data.
# - Sanity check: pull one batch and print its shape / show an image with
#   matplotlib to confirm labels line up with what you expect.


# ---------------------------------------------------------------------------
# STEP 4: Define a simple CNN model
# ---------------------------------------------------------------------------
# - Subclass torch.nn.Module.
# - Start small: 2-3 convolutional layers (nn.Conv2d) each followed by a
#   nonlinearity (nn.ReLU) and downsampling (nn.MaxPool2d), then flatten
#   and feed into one or two nn.Linear layers ending in a single output
#   (a logit for "face" vs "no face").
# - Don't overengineer this — a small CNN is enough to learn the difference
#   between "has a face" and "doesn't" on a curated dataset.


# ---------------------------------------------------------------------------
# STEP 5: Choose a loss function and optimizer
# ---------------------------------------------------------------------------
# - Since this is binary classification, use
#   torch.nn.BCEWithLogitsLoss (pairs naturally with a single output logit).
# - Pick an optimizer, e.g. torch.optim.Adam, with a learning rate to start
#   experimenting with (e.g. 1e-3) — you'll tune this later.


# ---------------------------------------------------------------------------
# STEP 6: Write the training loop
# ---------------------------------------------------------------------------
# - Move the model to your device (CPU or GPU).
# - For each epoch:
#     - Loop over batches from the train DataLoader.
#     - Zero gradients, forward pass, compute loss, backward pass, optimizer
#       step.
#     - Track running loss so you can see it decreasing.
# - After each epoch, run a validation pass (no gradient updates!) and
#   compute accuracy on the val set so you can see whether the model is
#   actually generalizing.


# ---------------------------------------------------------------------------
# STEP 7: Evaluate and iterate
# ---------------------------------------------------------------------------
# - Look at validation accuracy/loss across epochs — is it improving, or
#   overfitting (train accuracy climbs, val accuracy stalls/drops)?
# - Try simple fixes: more data, data augmentation (random flips/crops via
#   torchvision.transforms), a slightly deeper network, weight decay, or
#   dropout.
# - Save the trained weights with torch.save(model.state_dict(), ...) once
#   you're happy with it.


# ---------------------------------------------------------------------------
# STEP 8: Run inference on a new image
# ---------------------------------------------------------------------------
# - Load a single image file, apply the same transforms used in training
#   (same resize/normalize — this matters!).
# - Add a batch dimension, run it through the model in eval mode with
#   torch.no_grad(), apply a sigmoid to the logit, and threshold at 0.5 to
#   get a yes/no "face detected" answer.
# - Try it on photos you take yourself to see how well it generalizes.


# ---------------------------------------------------------------------------
# GOING FURTHER: from "is there a face" to "where is the face"
# ---------------------------------------------------------------------------
# Once step 8 works, you can grow this into real bounding-box detection:
#   - Easiest path: don't train a detector from scratch — use a pretrained
#     one. Options include torchvision's provided detection models, or
#     well-known face-specific detectors (e.g. MTCNN, RetinaFace) which
#     have PyTorch implementations you can pip install and call directly.
#   - Harder/educational path: implement a sliding-window approach — run
#     your Step 4 classifier over overlapping crops of a larger image at
#     multiple scales, and keep the crops where it's confident a face is
#     present. This is slow but teaches you what real detectors optimize
#     away.
#   - Proper path: learn about anchor-based detectors (e.g. how SSD/YOLO
#     work) and either implement a minimal version or fine-tune an
#     existing torchvision detection model on a face dataset with
#     bounding-box annotations (e.g. WIDER FACE).
