# Object Counting (Hailo Apps)

This example shows how to implement an *object counting* architecture using the Hailo Apps framework:

- **Detection** via `hailonet` + postprocess (`hailofilter`)
- **Tracking** via `hailotracker` to attach a stable `HAILO_UNIQUE_ID`
- **Counting** in a Python pad-probe callback by watching each tracked object's centroid crossing a **line** or entering/leaving a **polygon**.

## Files

- [object_counting_pipeline.py](object_counting_pipeline.py): builds the pipeline (source → inference → tracker → callback → display)
- [object_counting.py](object_counting.py): callback logic (centroids + IN/OUT counters) and optional OpenCV preview (`--use-frame`)

## Run

Count people crossing a **line** (default region):

```bash
python -m hailo_apps.hailo_app_python.apps.object_counting.object_counting --input <video_or_camera> --use-frame
```

Count inside a **polygon** region:

```bash
python -m hailo_apps.hailo_app_python.apps.object_counting.object_counting \
  --input <video_or_camera> \
  --use-frame \
  --region "20,400;1080,400;1080,360;20,360"
```

Count only specific labels (example):

```bash
python -m hailo_apps.hailo_app_python.apps.object_counting.object_counting \
  --input <video_or_camera> \
  --use-frame \
  --count-labels "person,car"
```

Track across **all** classes (instead of a single `class-id`):

```bash
python -m hailo_apps.hailo_app_python.apps.object_counting.object_counting \
  --input <video_or_camera> \
  --use-frame \
  --tracker-class-id -1
```

## Notes

- For accurate pixel regions, use `--use-frame` so the callback also extracts frames for visualization.
- Counting relies on `hailotracker` attaching `HAILO_UNIQUE_ID`. Without it, stable per-object counting is not possible.
