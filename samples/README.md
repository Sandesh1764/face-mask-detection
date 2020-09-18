# Sample media

Curated inputs and model outputs for the README and quick visual verification.

| Path | Description |
|------|-------------|
| `inputs/images/` | Public test stills (mask / no mask / mixed) |
| `outputs/images/` | Annotated predictions on those stills + demo video frames |
| `outputs/videos/` | Short annotated clips (~5 s) |

Regenerate still predictions:

```bash
python detect_mask_image.py --image samples/inputs/images/pic1.jpeg
```

Regenerate video clip:

```bash
python detect_mask_video_file.py --input your_video.mp4 --output samples/outputs/videos/out.mp4
```
