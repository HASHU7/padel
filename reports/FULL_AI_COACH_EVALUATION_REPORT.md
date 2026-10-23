# 🎾 Padel AI Coach: Complete Game, Technique & Foul Evaluation Report

## 1. Executive Summary

* **Tournament & Dataset:** World Padel Tour Barcelona Master Final (womens_final)
* **Total Ground-Truth Shots Evaluated:** 919
* **Complete Match Points Segmented:** 32 points across 4 games
* **Total Shots Analyzed in Match:** 450
* **Final Match Score:** `Set 1: 2 - 1`
* **Total FIP Rule Infractions / Fouls Detected:** `40`
* **Shot Classification Accuracy:** `46.35%`
* **High-Confidence Prediction Accuracy:** `55.91%`

## 2. Official FIP Rule Infraction & Foul Detection Audit

| Foul / Rule Infraction Type | Official FIP Rule | Detected Occurrences | Severity | Primary Coaching Correction |
| :--- | :--- | :--- | :--- | :--- |
| **High Contact Serve Fault (Saque sobre la Cintura)** | FIP Rule 6 (The Service) § c | `17` | `SERVICE_FAULT` | Allow ball to drop below hip height after bounce. Keep racket head below wrist at impact to impart legal heavy slice. |
| **Direct Glass Out (Pared Directa sin Bote)** | FIP Rule 13 (Lost Point) § c | `12` | `POINT_LOSS` | Add topspin or heavy slice to dip the ball into the court. Do not hit flat drives from deep defensive positions. |
| **Foot Fault on Service (Falta de Pie)** | FIP Rule 6 (The Service) § a/b | `3` | `SERVICE_FAULT` | Anchor rear foot firmly 30cm behind the line. Toss ball slightly back rather than chasing it forward into the court. |
| **Double Bounce (Doble Bote)** | FIP Rule 13 (Lost Point) § a | `8` | `POINT_LOSS` | React faster to low chiquitas or drop shots. Anticipate the bounce by reading opponent's racket angle. |

## 3. Shot Classification Performance Metrics

| Shot Category | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Backhand** | 0.43 | 0.46 | 0.44 | 278 |
| **Dropshot** | 0.00 | 0.00 | 0.00 | 5 |
| **Forehand** | 0.42 | 0.49 | 0.45 | 284 |
| **Other** | 0.06 | 0.01 | 0.02 | 67 |
| **Serve** | 0.64 | 0.58 | 0.60 | 73 |
| **Smash** | 0.57 | 0.54 | 0.56 | 212 |

## 4. Player-by-Player Biomechanical Audits & 4-Week Development Roadmaps

### Player 0 (Near Court Left)
* **Team:** `Team 1 (Near Court)` | **Coach Grade:** `B` | **Overall Rating:** `78/100`
* **Total Shots Hit:** 97 | **Fouls Committed:** 12
* **Shot Breakdown:** `{'Backhand': 34, 'Forehand': 31, 'Smash': 19, 'Other': 7, 'Serve': 4, 'Dropshot': 2}`
* **Average Knee Flexion:** `138.2°`
* **Average Ball Exit Speed:** `65.6 km/h`
* **Court Zone Distribution:** Net: 30.9% | Baseline: 13.4% | Transition: 55.7%

**Primary Technical & Tactical Flaws:**
- **Caught in No-Man's Land (Transition Zone)** (50x occurrences - Tactical Court Positioning): Shot executed 5.9m from net. Lingering in transition exposes you to low dipping balls at your feet.
  * *Actionable Coaching Cue:* `Zone Discipline: Never stay in transition! If advancing, sprint to the net. If pushed back, retreat behind the service line.`
- **Over-Extended Early Contact** (28x occurrences - Contact Timing): Reaching too far in front; lost rotational power from core.
  * *Actionable Coaching Cue:* `Let the ball come slightly closer to your strike zone before unleashing the kinetic chain.`
- **Stiff Knees / Bending at Waist** (27x occurrences - Footwork & Balance): Knee flexion was 176.2° on a low ball. Bending from the waist causes errors into the net cord.
  * *Actionable Coaching Cue:* `Squat down with a wide athletic base. Lower your hips so your eyes are level with the ball trajectory.`
- **Cramped Overhead Reach ('T-Rex Arm')** (12x occurrences - Overhead Extension): Contact height was too low. Overhead shots require high vertical reach for steep downward penetration.
  * *Actionable Coaching Cue:* `Point at the ball with your non-dominant hand. Reach to full extension at 12 o'clock above your dominant shoulder.`

**Assigned Certified Training Drills:**
- 🎾 **Net-to-Base Transition Sprints** (Target: Court Positioning & Eliminating Transition Flaws)
  * *Objective:* Instantly transition between attack and defense without stalling in the danger zone.
  * *Setup:* Touch net tape with racket, sprint backward behind service line to defend lob, then sprint immediately forward to re-take net.
  * *Prescription:* `6 continuous rounds of 5 transitions.`
  * *Key Coaching Focus:* `Never stop running until both feet are completely behind the service line or inside the kitchen zone.`
- 🎾 **Low Tray Knee Flexion Drill** (Target: Center of Gravity & Squat Depth on Defense)
  * *Objective:* Bend knees to 115° on low balls without hinging at the waist.
  * *Setup:* Place mini-cones at the baseline. Player must touch cone with knee before executing low chiquitas or wall returns.
  * *Prescription:* `5 sets of 12 balls with rapid lateral recovery.`
  * *Key Coaching Focus:* `Drop hips straight down like an elevator, not bending forward like a crane.`
- 🎾 **Apex Reach & Vibora Extension Drill** (Target: Overhead Apex Reach (Smash & Bandeja))
  * *Objective:* Achieve full vertical extension for steep trajectory into opponent's glass.
  * *Setup:* High lobs fed from deep court. Hit smashes targeting opponent's corner glass with full vertical extension.
  * *Prescription:* `3 sets of 20 overhead strikes.`
  * *Key Coaching Focus:* `Keep non-dominant hand pointing at the ball until right before impact; extend triceps fully.`
- 🎾 **Service Line Footwork Discipline Drill** (Target: Eliminating Service Foot Faults)
  * *Objective:* Maintain legal foot placement behind service line and center mark on serve.
  * *Setup:* Place chalk or training tape 15cm behind service line. Server must keep rear foot anchored behind tape through contact.
  * *Prescription:* `50 practice serves focusing strictly on legal foot placement.`
  * *Key Coaching Focus:* `Toss ball slightly backward so momentum carries into ball without feet stepping forward early.`

**Structured 4-Week Development Curriculum:**
- **Week 1: Footwork & Center of Gravity**: Eliminating stiff knees and lowering hips on low defense
  * Core Drill: `Low Tray Knee Flexion Drill (5 sets x 12 reps)` | Target: `Knee flexion angle below 135° on 90% of defensive balls`
- **Week 2: Contact Point Timing & Unit Turn**: Early shoulder coil and hitting 25cm ahead of lead hip
  * Core Drill: `Wall-Rebound Early Prep Drill (4 sets x 15 reps)` | Target: `Zero late contact points recorded off back glass`
- **Week 3: Overhead Dominance & Power**: Full vertical extension on smash and bandeja
  * Core Drill: `Apex Reach & Vibora Extension Drill (3 sets x 20 reps)` | Target: `Exit ball speed over 105 km/h with steep downward angle`
- **Week 4: Tactical Zone Discipline & Foul Eradication**: Zero time wasted in No-Man's land and clean footwork on serve
  * Core Drill: `Net-to-Base Transition Sprints (6 rounds x 5 reps)` | Target: `Transition zone occupancy under 15% and zero match foot faults`

---

### Player 1 (Far Court Left)
* **Team:** `Team 2 (Far Court)` | **Coach Grade:** `B` | **Overall Rating:** `77/100`
* **Total Shots Hit:** 142 | **Fouls Committed:** 11
* **Shot Breakdown:** `{'Forehand': 50, 'Smash': 46, 'Backhand': 26, 'Serve': 11, 'Other': 9}`
* **Average Knee Flexion:** `147.5°`
* **Average Ball Exit Speed:** `73.6 km/h`
* **Court Zone Distribution:** Net: 2.1% | Baseline: 71.1% | Transition: 26.8%

**Primary Technical & Tactical Flaws:**
- **Caught in No-Man's Land (Transition Zone)** (46x occurrences - Tactical Court Positioning): Shot executed 5.3m from net. Lingering in transition exposes you to low dipping balls at your feet.
  * *Actionable Coaching Cue:* `Zone Discipline: Never stay in transition! If advancing, sprint to the net. If pushed back, retreat behind the service line.`
- **Stiff Knees / Bending at Waist** (38x occurrences - Footwork & Balance): Knee flexion was 157.0° on a low ball. Bending from the waist causes errors into the net cord.
  * *Actionable Coaching Cue:* `Squat down with a wide athletic base. Lower your hips so your eyes are level with the ball trajectory.`
- **Late Contact Point** (26x occurrences - Contact Timing): Ball struck 152.6cm behind the lead hip axis. Results in spraying wide or hitting the side fence.
  * *Actionable Coaching Cue:* `Unit Turn Early: Start your shoulder coil as soon as the ball leaves the opponent's racket. Meet the ball 25cm in front of your lead hip.`
- **Cramped Overhead Reach ('T-Rex Arm')** (24x occurrences - Overhead Extension): Contact height was too low. Overhead shots require high vertical reach for steep downward penetration.
  * *Actionable Coaching Cue:* `Point at the ball with your non-dominant hand. Reach to full extension at 12 o'clock above your dominant shoulder.`

**Assigned Certified Training Drills:**
- 🎾 **Net-to-Base Transition Sprints** (Target: Court Positioning & Eliminating Transition Flaws)
  * *Objective:* Instantly transition between attack and defense without stalling in the danger zone.
  * *Setup:* Touch net tape with racket, sprint backward behind service line to defend lob, then sprint immediately forward to re-take net.
  * *Prescription:* `6 continuous rounds of 5 transitions.`
  * *Key Coaching Focus:* `Never stop running until both feet are completely behind the service line or inside the kitchen zone.`
- 🎾 **Low Tray Knee Flexion Drill** (Target: Center of Gravity & Squat Depth on Defense)
  * *Objective:* Bend knees to 115° on low balls without hinging at the waist.
  * *Setup:* Place mini-cones at the baseline. Player must touch cone with knee before executing low chiquitas or wall returns.
  * *Prescription:* `5 sets of 12 balls with rapid lateral recovery.`
  * *Key Coaching Focus:* `Drop hips straight down like an elevator, not bending forward like a crane.`
- 🎾 **Wall-Rebound Early Prep Drill** (Target: Contact Point Timing (Forehand & Backhand))
  * *Objective:* Eliminate late contact off the back wall and strike 25cm in front of lead hip.
  * *Setup:* Place 2 cones 25cm ahead of your lead foot. Feed rapid balls off the back glass; racket must meet ball precisely between cones.
  * *Prescription:* `4 sets of 15 repetitions alternating crosscourt and down-the-line.`
  * *Key Coaching Focus:* `Rotate your shoulders early so the racket is already back before the ball hits the glass.`
- 🎾 **Apex Reach & Vibora Extension Drill** (Target: Overhead Apex Reach (Smash & Bandeja))
  * *Objective:* Achieve full vertical extension for steep trajectory into opponent's glass.
  * *Setup:* High lobs fed from deep court. Hit smashes targeting opponent's corner glass with full vertical extension.
  * *Prescription:* `3 sets of 20 overhead strikes.`
  * *Key Coaching Focus:* `Keep non-dominant hand pointing at the ball until right before impact; extend triceps fully.`

**Structured 4-Week Development Curriculum:**
- **Week 1: Footwork & Center of Gravity**: Eliminating stiff knees and lowering hips on low defense
  * Core Drill: `Low Tray Knee Flexion Drill (5 sets x 12 reps)` | Target: `Knee flexion angle below 135° on 90% of defensive balls`
- **Week 2: Contact Point Timing & Unit Turn**: Early shoulder coil and hitting 25cm ahead of lead hip
  * Core Drill: `Wall-Rebound Early Prep Drill (4 sets x 15 reps)` | Target: `Zero late contact points recorded off back glass`
- **Week 3: Overhead Dominance & Power**: Full vertical extension on smash and bandeja
  * Core Drill: `Apex Reach & Vibora Extension Drill (3 sets x 20 reps)` | Target: `Exit ball speed over 105 km/h with steep downward angle`
- **Week 4: Tactical Zone Discipline & Foul Eradication**: Zero time wasted in No-Man's land and clean footwork on serve
  * Core Drill: `Net-to-Base Transition Sprints (6 rounds x 5 reps)` | Target: `Transition zone occupancy under 15% and zero match foot faults`

---

### Player 2 (Far Court Right)
* **Team:** `Team 2 (Far Court)` | **Coach Grade:** `B+` | **Overall Rating:** `81/100`
* **Total Shots Hit:** 131 | **Fouls Committed:** 4
* **Shot Breakdown:** `{'Backhand': 41, 'Smash': 38, 'Forehand': 32, 'Serve': 11, 'Other': 9}`
* **Average Knee Flexion:** `147.1°`
* **Average Ball Exit Speed:** `82.4 km/h`
* **Court Zone Distribution:** Net: 1.5% | Baseline: 79.4% | Transition: 19.1%

**Primary Technical & Tactical Flaws:**
- **Over-Extended Early Contact** (29x occurrences - Contact Timing): Reaching too far in front; lost rotational power from core.
  * *Actionable Coaching Cue:* `Let the ball come slightly closer to your strike zone before unleashing the kinetic chain.`
- **Caught in No-Man's Land (Transition Zone)** (27x occurrences - Tactical Court Positioning): Shot executed 5.5m from net. Lingering in transition exposes you to low dipping balls at your feet.
  * *Actionable Coaching Cue:* `Zone Discipline: Never stay in transition! If advancing, sprint to the net. If pushed back, retreat behind the service line.`
- **Stiff Knees / Bending at Waist** (26x occurrences - Footwork & Balance): Knee flexion was 152.2° on a low ball. Bending from the waist causes errors into the net cord.
  * *Actionable Coaching Cue:* `Squat down with a wide athletic base. Lower your hips so your eyes are level with the ball trajectory.`
- **Cramped Overhead Reach ('T-Rex Arm')** (20x occurrences - Overhead Extension): Contact height was too low. Overhead shots require high vertical reach for steep downward penetration.
  * *Actionable Coaching Cue:* `Point at the ball with your non-dominant hand. Reach to full extension at 12 o'clock above your dominant shoulder.`

**Assigned Certified Training Drills:**
- 🎾 **Net-to-Base Transition Sprints** (Target: Court Positioning & Eliminating Transition Flaws)
  * *Objective:* Instantly transition between attack and defense without stalling in the danger zone.
  * *Setup:* Touch net tape with racket, sprint backward behind service line to defend lob, then sprint immediately forward to re-take net.
  * *Prescription:* `6 continuous rounds of 5 transitions.`
  * *Key Coaching Focus:* `Never stop running until both feet are completely behind the service line or inside the kitchen zone.`
- 🎾 **Low Tray Knee Flexion Drill** (Target: Center of Gravity & Squat Depth on Defense)
  * *Objective:* Bend knees to 115° on low balls without hinging at the waist.
  * *Setup:* Place mini-cones at the baseline. Player must touch cone with knee before executing low chiquitas or wall returns.
  * *Prescription:* `5 sets of 12 balls with rapid lateral recovery.`
  * *Key Coaching Focus:* `Drop hips straight down like an elevator, not bending forward like a crane.`
- 🎾 **Apex Reach & Vibora Extension Drill** (Target: Overhead Apex Reach (Smash & Bandeja))
  * *Objective:* Achieve full vertical extension for steep trajectory into opponent's glass.
  * *Setup:* High lobs fed from deep court. Hit smashes targeting opponent's corner glass with full vertical extension.
  * *Prescription:* `3 sets of 20 overhead strikes.`
  * *Key Coaching Focus:* `Keep non-dominant hand pointing at the ball until right before impact; extend triceps fully.`

**Structured 4-Week Development Curriculum:**
- **Week 1: Footwork & Center of Gravity**: Eliminating stiff knees and lowering hips on low defense
  * Core Drill: `Low Tray Knee Flexion Drill (5 sets x 12 reps)` | Target: `Knee flexion angle below 135° on 90% of defensive balls`
- **Week 2: Contact Point Timing & Unit Turn**: Early shoulder coil and hitting 25cm ahead of lead hip
  * Core Drill: `Wall-Rebound Early Prep Drill (4 sets x 15 reps)` | Target: `Zero late contact points recorded off back glass`
- **Week 3: Overhead Dominance & Power**: Full vertical extension on smash and bandeja
  * Core Drill: `Apex Reach & Vibora Extension Drill (3 sets x 20 reps)` | Target: `Exit ball speed over 105 km/h with steep downward angle`
- **Week 4: Tactical Zone Discipline & Foul Eradication**: Zero time wasted in No-Man's land and clean footwork on serve
  * Core Drill: `Net-to-Base Transition Sprints (6 rounds x 5 reps)` | Target: `Transition zone occupancy under 15% and zero match foot faults`

---

### Player 3 (Near Court Right)
* **Team:** `Team 1 (Near Court)` | **Coach Grade:** `B` | **Overall Rating:** `75/100`
* **Total Shots Hit:** 80 | **Fouls Committed:** 13
* **Shot Breakdown:** `{'Forehand': 34, 'Backhand': 20, 'Smash': 12, 'Other': 8, 'Serve': 6}`
* **Average Knee Flexion:** `140.7°`
* **Average Ball Exit Speed:** `60.0 km/h`
* **Court Zone Distribution:** Net: 30.0% | Baseline: 23.8% | Transition: 46.2%

**Primary Technical & Tactical Flaws:**
- **Stiff Knees / Bending at Waist** (33x occurrences - Footwork & Balance): Knee flexion was 159.5° on a low ball. Bending from the waist causes errors into the net cord.
  * *Actionable Coaching Cue:* `Squat down with a wide athletic base. Lower your hips so your eyes are level with the ball trajectory.`
- **Caught in No-Man's Land (Transition Zone)** (31x occurrences - Tactical Court Positioning): Shot executed 5.8m from net. Lingering in transition exposes you to low dipping balls at your feet.
  * *Actionable Coaching Cue:* `Zone Discipline: Never stay in transition! If advancing, sprint to the net. If pushed back, retreat behind the service line.`
- **Over-Extended Early Contact** (17x occurrences - Contact Timing): Reaching too far in front; lost rotational power from core.
  * *Actionable Coaching Cue:* `Let the ball come slightly closer to your strike zone before unleashing the kinetic chain.`
- **Late Contact Point** (17x occurrences - Contact Timing): Ball struck 124.8cm behind the lead hip axis. Results in spraying wide or hitting the side fence.
  * *Actionable Coaching Cue:* `Unit Turn Early: Start your shoulder coil as soon as the ball leaves the opponent's racket. Meet the ball 25cm in front of your lead hip.`

**Assigned Certified Training Drills:**
- 🎾 **Low Tray Knee Flexion Drill** (Target: Center of Gravity & Squat Depth on Defense)
  * *Objective:* Bend knees to 115° on low balls without hinging at the waist.
  * *Setup:* Place mini-cones at the baseline. Player must touch cone with knee before executing low chiquitas or wall returns.
  * *Prescription:* `5 sets of 12 balls with rapid lateral recovery.`
  * *Key Coaching Focus:* `Drop hips straight down like an elevator, not bending forward like a crane.`
- 🎾 **Net-to-Base Transition Sprints** (Target: Court Positioning & Eliminating Transition Flaws)
  * *Objective:* Instantly transition between attack and defense without stalling in the danger zone.
  * *Setup:* Touch net tape with racket, sprint backward behind service line to defend lob, then sprint immediately forward to re-take net.
  * *Prescription:* `6 continuous rounds of 5 transitions.`
  * *Key Coaching Focus:* `Never stop running until both feet are completely behind the service line or inside the kitchen zone.`
- 🎾 **Wall-Rebound Early Prep Drill** (Target: Contact Point Timing (Forehand & Backhand))
  * *Objective:* Eliminate late contact off the back wall and strike 25cm in front of lead hip.
  * *Setup:* Place 2 cones 25cm ahead of your lead foot. Feed rapid balls off the back glass; racket must meet ball precisely between cones.
  * *Prescription:* `4 sets of 15 repetitions alternating crosscourt and down-the-line.`
  * *Key Coaching Focus:* `Rotate your shoulders early so the racket is already back before the ball hits the glass.`
- 🎾 **Service Line Footwork Discipline Drill** (Target: Eliminating Service Foot Faults)
  * *Objective:* Maintain legal foot placement behind service line and center mark on serve.
  * *Setup:* Place chalk or training tape 15cm behind service line. Server must keep rear foot anchored behind tape through contact.
  * *Prescription:* `50 practice serves focusing strictly on legal foot placement.`
  * *Key Coaching Focus:* `Toss ball slightly backward so momentum carries into ball without feet stepping forward early.`

**Structured 4-Week Development Curriculum:**
- **Week 1: Footwork & Center of Gravity**: Eliminating stiff knees and lowering hips on low defense
  * Core Drill: `Low Tray Knee Flexion Drill (5 sets x 12 reps)` | Target: `Knee flexion angle below 135° on 90% of defensive balls`
- **Week 2: Contact Point Timing & Unit Turn**: Early shoulder coil and hitting 25cm ahead of lead hip
  * Core Drill: `Wall-Rebound Early Prep Drill (4 sets x 15 reps)` | Target: `Zero late contact points recorded off back glass`
- **Week 3: Overhead Dominance & Power**: Full vertical extension on smash and bandeja
  * Core Drill: `Apex Reach & Vibora Extension Drill (3 sets x 20 reps)` | Target: `Exit ball speed over 105 km/h with steep downward angle`
- **Week 4: Tactical Zone Discipline & Foul Eradication**: Zero time wasted in No-Man's land and clean footwork on serve
  * Core Drill: `Net-to-Base Transition Sprints (6 rounds x 5 reps)` | Target: `Transition zone occupancy under 15% and zero match foot faults`

---

## 5. Sample Game & Point Reconstructions

### Point #1 (Game 1 • Score: 0 - 15)
* **Duration:** 14.3s (10 shots @ 42.0 shots/min) | **Peak Speed:** 174.0 km/h
* **Outcome:** `FORCED_DEFENSIVE_ERROR` | **Winner:** `Team 2 (Far Court)`
* **Tactical Analysis:** Team 2 (Far Court) forced an error through continuous pressure (Smash).
* **Shot Sequence:** `Serve (Player 1) ➔ Forehand (Player 3) ➔ Smash (Player 2) ➔ Backhand (Player 0) ➔ Smash (Player 1) ➔ Backhand (Player 0) ➔ Smash (Player 2) ➔ Other (Player 0) ➔ Smash (Player 2) ➔ Other (Player 3)`

### Point #2 (Game 1 • Score: 0 - 30)
* **Duration:** 9.57s (10 shots @ 62.7 shots/min) | **Peak Speed:** 129.1 km/h
* **Outcome:** `UNFORCED_ERROR` | **Winner:** `Team 2 (Far Court)`
* **Tactical Analysis:** Team 1 (Near Court) committed an unforced error while trapped in the transition zone.
* **Shot Sequence:** `Serve (Player 2) ➔ Backhand (Player 0) ➔ Backhand (Player 1) ➔ Forehand (Player 0) ➔ Forehand (Player 1) ➔ Backhand (Player 0) ➔ Forehand (Player 2) ➔ Forehand (Player 0) ➔ Smash (Player 1) ➔ Other (Player 3)`

### Point #3 (Game 1 • Score: 15 - 30)
* **Duration:** 8.3s (7 shots @ 50.6 shots/min) | **Peak Speed:** 136.5 km/h
* **Outcome:** `UNFORCED_ERROR` | **Winner:** `Team 1 (Near Court)`
* **Tactical Analysis:** Team 2 (Far Court) committed an unforced error while trapped in the transition zone.
* **Shot Sequence:** `Serve (Player 1) ➔ Forehand (Player 3) ➔ Smash (Player 1) ➔ Forehand (Player 3) ➔ Forehand (Player 1) ➔ Smash (Player 2) ➔ Backhand (Player 2)`

### Point #4 (Game 1 • Score: 30 - 30)
* **Duration:** 20.63s (15 shots @ 43.6 shots/min) | **Peak Speed:** 180.0 km/h
* **Outcome:** `UNFORCED_ERROR` | **Winner:** `Team 1 (Near Court)`
* **Tactical Analysis:** Team 2 (Far Court) committed an unforced error while trapped in the transition zone.
* **Shot Sequence:** `Serve (Player 2) ➔ Backhand (Player 0) ➔ Smash (Player 1) ➔ Forehand (Player 1) ➔ Smash (Player 1) ➔ Forehand (Player 0) ➔ Backhand (Player 2) ➔ Smash (Player 2) ➔ Forehand (Player 1) ➔ Backhand (Player 0) ➔ Forehand (Player 2) ➔ Smash (Player 0) ➔ Backhand (Player 2) ➔ Forehand (Player 0) ➔ Backhand (Player 1)`

### Point #5 (Game 1 • Score: 30 - 40)
* **Duration:** 13.97s (9 shots @ 38.7 shots/min) | **Peak Speed:** 180.0 km/h
* **Outcome:** `FOUL: Direct Glass Out (Pared Directa sin Bote)` | **Winner:** `Team 2 (Far Court)`
* **Tactical Analysis:** Team 1 (Near Court) conceded the point due to a rule infraction: Direct Glass Out (Pared Directa sin Bote) (FIP Rule 13 (Lost Point) § c). Stroke hit opponent's perimeter glass directly on the full without first bouncing on the court floor.
* **Shot Sequence:** `Serve (Player 1) ➔ Forehand (Player 3) ➔ Smash (Player 2) ➔ Forehand (Player 1) ➔ Forehand (Player 3) ➔ Smash (Player 1) ➔ Forehand (Player 3) ➔ Smash (Player 2) ➔ Dropshot (Player 0)`
