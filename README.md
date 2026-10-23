# Padel AI Coach

Full-match padel video analysis: every shot, every point and the score, FIP rule checks, per-player and per-team statistics, weaknesses for your own team and the opponents, a scouting game plan, and training plans with drills. It runs locally as a web app. Model training happens on Google Colab only.

```
python -m padelcoach serve          # then open http://127.0.0.1:8000
```

## How it works

```
video ─► perception ─► identities ─► ball events ─► shots ─► points & score ─► FIP rules ─► analytics ─► coach
         (pose, ball,    (4 players,   (hit, bounce,  (type,     (serve detection,  (serve faults,  (players, teams,
          court)          teams, sides)  wall)         context)   winner, scoring)   walls, net…)    weaknesses, plans)
```

| Stage | Method |
|---|---|
| Court | Floor homography from court landmarks. The WPT broadcast preset is verified at 1.2 cm mean error. Other cameras are calibrated by clicking in the app. Positions use the **feet**, because the floor mapping is only valid on the floor. |
| Players | Court-space tracking into short tracklets, then **kit-colour clustering** for teams. This follows teams through **changeovers**. Partners are split into left/right player by side preference, height and appearance. |
| Ball events | Impulses in the 2D ball trajectory become a *hit* (within racket reach of a player), a *bounce* (downward→upward motion with the floor projection inside the court) or a *wall* impact (projection at or outside the glass or mesh). Padel's rule that shots alternate between teams removes ball-bouncing and partner "touches". |
| Shots | Serve detection uses the serve formation (FIP Rule 6): server deep and still, ball bounced first, receiver on the diagonal, partner at the net. Stroke side comes from the contact point in the player's own frame plus estimated handedness. Subtypes: drive, volley, off the glass, lob, smash, bandeja, víbora, drop shot, chiquita, return. An optional Colab-trained temporal model refines the stroke. |
| Points | A rally runs from a serve until play stops. Balls fed between points are excluded. The winner comes from what the ball did after the last shot, following FIP Rule 13 (into the net or own side, direct to the opponents' glass, unreturned bounce). The **broadcast scoreboard** (OCR on Colab) overrides this, and any point can be corrected in the app. Scoring handles advantage or golden point, tie-break, server rotation and changeovers. |
| Rules | FIP Rules of Padel (2026): serve foot fault, contact above the waist, no bounce, serve outside the box, serve onto the mesh, double fault, direct wall, double bounce, net invasion. Each check has explicit measurement margins and a confidence (*likely* / *possible* / *review*). |
| Analytics | Shot mix and error, winner and efficiency rates; zone time, heatmaps, distance covered; pair positioning (both at the net, split, gaps); serve and return; break points; lob defence; biomechanics (knee flexion on low balls, overhead extension, contact height and spacing). |
| Coaching | Weaknesses are measured against a **pro benchmark** built from the two WPT finals, with small-sample shrinkage, an impact estimate in points and a confidence. Strengths, a 4-week plan with drills, and a scouting game plan ("how to beat them") are generated per team. |

## Measured accuracy on UNSEEN matches (honest, cross-match)

Every model used here (pose, ball, shot type) was trained on the *other* final only; the match being
analysed was never seen in training. Raw video in, labels used only for scoring.

| Unseen match | Women's final | Men's final |
|---|---|---|
| Players found (recall / precision) | 99.9 % / 99.3 % | 98.8 % / 98.4 % |
| Body keypoints correct (PCK@0.1) | 96.1 % | 95.7 % |
| Ball found within 10 px (precision / recall) | 88.3 % / 89.1 % | 89.8 % / 84.2 % |
| Shots detected (F1) | 0.913 | 0.877 |
| Stroke type correct (6 classes) | 80.7 % (86.9 % excl. "Other") | 82.0 % (87.1 % excl. "Other") |
| Points confirmed by the broadcast scoreboard (OCR) | 59 | 75 |

- Ball tracking keeps the top-5 detections per frame and links them over time ("link_strict"); this
  method was chosen among 5 by F1 on these two matches (mild selection on test data).
- Fine-tuned pose vs off-the-shelf YOLO11s-pose on the unseen match: keypoint mAP50-95 0.67 vs 0.47-0.62.
- Limits: both finals are the same venue, camera and broadcast. Other courts/cameras need training
  data from other venues (Roboflow Universe padel sets, CC BY 4.0). The first set score now matches the
  broadcast; later sets can still drift when games are missed by the scoreboard OCR.

### Unseen venues (4 held-out WPT clips, 2,290 frames; never used for training or tuning)

| Model | Before (Barcelona-only) | Now (multi-venue) |
|---|---|---|
| Ball (P / R / F1, 10 px) | 81 % / 77 % / 0.79 | 99 % / 99 % / 0.99 |
| Players (pose model, F1) | 0.70 (off-the-shelf: 0.80) | 0.98 |
| Body keypoints, Barcelona held-out frames (mAP50-95) | 0.92 (same match it trained on; 0.67 on an unseen match) | 0.82 |

Caveats: some held-out clips come from the same tournaments as training clips; labelled frames mostly show
a visible ball (full videos will score lower); keypoints on other venues have no ground truth (pseudo-labelled).
External data: kaggle ichimarugin/padel-ball-dataset (licence unknown - check before commercial use).
Shot model: ball-feature dropout + 50 % of josmazna/padelpose; broadcast unseen match 78-84 %.

Cross-match training/evaluation jobs: `kaggle/make_cv_kernels.py` (results in `kaggle_outputs/cv/`).

## Workflows

**1. Labelled PadelTracker100 matches** (`2022_BCN_FinalF_1.mp4`, `2022_BCN_FinalM_1.mp4`): analysed from the annotations in the *labels* source. This works without any model.

**2. Any other match video**

1. *(once)* Run `python -m padelcoach export-training`, then upload `colab_bundle/padel_bundle.zip` and the two finals to `MyDrive/padel/`.
2. *(once)* Run **`colab/padel_train_models.ipynb`** on a T4. It trains the ball detector, the pose model and the shot classifier, exports ONNX, and its downloads go into `models/`.
3. *(per match)* Run **`colab/padel_gpu_inference.ipynb`** on the video, about 30–60 min on a T4. It produces `<video>.npz` (players, poses, ball) and `<video>.json` (scoreboard).
4. In the app, on the video's card: **Import Colab file** (both files), choose source **import**, and press **Analyse**. If the camera isn't the WPT broadcast view, use **Calibrate court** first.

With the ONNX models in `models/`, short clips can also run fully locally (source *video*). A whole match on this laptop's CPU would take many hours, which is why per-match inference is done on Colab.

**3. Your team vs opponents**: open the match and choose **Our team**, or set names under *Settings*. You get your own weaknesses and plan, plus the opponents' scouting report.

## CLI

```
python -m padelcoach videos                 # videos + available sources
python -m padelcoach analyze VIDEO [--mode auto|labels|import|video]
python -m padelcoach reanalyze VIDEO        # re-run analytics with saved settings (fast)
python -m padelcoach benchmark              # rebuild the pro benchmark from the labelled finals
python -m padelcoach export-training        # Colab bundle
python -m unittest discover -s tests        # unit tests
```

## Limits

- **Single broadcast camera, 2D.** The ball's height isn't measured. Speeds are averages from contact to bounce, and contact heights are estimates calibrated to an average player height.
- **Without the scoreboard, point winners are inferred.** About half are clear-cut from the ball's behaviour; the rest use shot-quality priors. Correct them in **Points & score** and every statistic updates. Importing the Colab scoreboard file makes them authoritative.
- **Rule checks are cautious but not referee-grade.** "Possible" means watch the clip, which the app links to.
- **Ball labels in the men's final only cover the first ~11 minutes.** In *labels* mode only that part has shots. A Colab ball model removes this limit.
- **Ball needs the Colab model.** The classical ball detector (no model) is unreliable on broadcast footage (precision/recall ~12 %/6 %). Raw-video analysis needs the Colab ball model.
- **PadelTracker100 alignment.** The men's pose/ball JSON starts 65 frames after its video. The loader corrects this; it was verified against the video pixels.

## Layout

```
padelcoach/            package (court, identity, events, shots, rally, scoring, fouls, analytics, drills, scoreboard,
                       perception, pipeline, report, app/ = FastAPI + web UI)
colab/                 training + GPU inference notebooks, scripts used by them
models/                ONNX models from Colab (padel_pose.onnx, padel_ball.onnx, padel_shot_classifier.onnx/.json)
labels/                PadelTracker100 annotations (CC-BY-4.0)
workspace/             videos/, calibrations/, detections/, scoreboard/, results/<video>/
tests/                 unit tests
src/, live_preview.html, match_data.js, reports/   previous prototype (superseded, kept for reference)
```

Licences: PadelTracker100 is CC-BY-4.0. Ultralytics YOLO, used in the Colab notebooks, is AGPL-3.0.

