from ultralytics import YOLO

if __name__ == "__main__":
  # Load your best trained segmentation model weights
  model = YOLO(r"D:\New width prj\runs\segment\train-6\weights\best.pt")

  # Run tracking with your custom tracker config file
  results = model.track(
      source=r"D:\New width prj\concrete31.mp4",  # Or your video/webcam
      show=True,
      save=True,
      persist=True,
      conf=0.4,
      iou=0.5,
      retina_masks=True,
      agnostic_nms=True,
  )