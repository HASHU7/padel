"""Drill library keyed by weakness category. Each drill has a measurable goal."""
from __future__ import annotations

DRILLS = {
    "backhand_volley": {
        "name": "Backhand volley wall series", "focus": "Compact backhand volley, early racket preparation",
        "setup": "Coach feeds from the opposite service line; player at 2.5 m from the net. 3 balls to the backhand body/side, 1 recovery split-step.",
        "dose": "5 x 12 balls, 45 s rest", "key": "Racket in front of the chest before the opponent strikes; block-and-push, no backswing."},
    "forehand_volley": {
        "name": "Forehand volley depth targets", "focus": "Volley depth and control",
        "setup": "Two target zones 1.5 m from the opponents' back glass. Feeds at medium pace to the forehand.",
        "dose": "4 x 15 balls", "key": "Short swing, slice under the ball, hold the finish."},
    "backhand_drive": {
        "name": "Backhand from the baseline under pressure", "focus": "Low centre of gravity and late-ball tolerance",
        "setup": "Feeds deep to the backhand corner; player recovers to the centre after each shot.",
        "dose": "4 x 12 balls", "key": "Turn shoulders early, contact in front of the front foot."},
    "forehand_drive": {
        "name": "Forehand cross-court consistency", "focus": "Forehand margin over the net",
        "setup": "Cross-court rally to a 2 x 2 m target, ball must clear the net by > 50 cm.",
        "dose": "5 x 2 min", "key": "Brush up, aim higher over the net, finish over the shoulder."},
    "off_wall": {
        "name": "Back-glass exits (salida de pared)", "focus": "Reading the rebound and patience",
        "setup": "Feeds into the back glass at varying heights; player lets the ball come off the glass and plays a deep cross-court or lob.",
        "dose": "6 x 8 balls (forehand + backhand)", "key": "Move away from the glass first, then follow the ball forward; contact at hip height."},
    "lob": {
        "name": "Defensive lob height & depth", "focus": "Lob quality from defence",
        "setup": "Player at the back glass, coach volleys from the net. Every ball must be lobbed over the coach landing in the last 2 m.",
        "dose": "4 x 10 lobs", "key": "Open racket face, lift through the ball, aim for the far corner."},
    "overhead": {
        "name": "Bandeja/vibora control box", "focus": "Overhead selection and control",
        "setup": "Lobs fed to the player at 4-6 m from the net; target box at the opponents' back-side glass. Smash only inside 4 m.",
        "dose": "5 x 10 overheads", "key": "Side-on early, non-hitting arm points at the ball, hit in front of the head."},
    "smash": {
        "name": "Smash put-away (por 3 / x4) progression", "focus": "Smash execution and choice",
        "setup": "Short lobs inside 3.5 m from the net; first flat smash to the glass, then x3 out of the side.",
        "dose": "4 x 8 smashes", "key": "Only full smash when the ball is inside the net zone; otherwise bandeja."},
    "transition": {
        "name": "Transition sprint to the net", "focus": "Not staying in no-man's land",
        "setup": "After a lob from the back, player sprints to the net line and plays a volley; coach immediately feeds the next ball.",
        "dose": "6 x 5 reps", "key": "Never stop between 4 and 7 m from the net: either close to 2-3 m or stay behind the service line."},
    "lob_defense": {
        "name": "Lob recovery (retreat + overhead)", "focus": "Defending lobs over the head",
        "setup": "Player at the net; coach lobs over. Player chooses: bandeja if reachable, else turns and plays off the glass.",
        "dose": "5 x 8 lobs", "key": "Turn sideways at once and cross-step back; decide early."},
    "serve": {
        "name": "Serve placement ladder", "focus": "First-serve percentage and placement",
        "setup": "Targets at the T and at the side glass corner; 10 serves per target, count serves in.",
        "dose": "3 x 20 serves", "key": "Slice, low bounce, keep contact below the waist and feet still."},
    "return": {
        "name": "Return to the server's feet / lob return", "focus": "Return consistency",
        "setup": "Coach serves; player alternates low cross-court return and lob return over the net player.",
        "dose": "4 x 12 returns", "key": "Short backswing, ball in front, first priority is in-court."},
    "footwork_low": {
        "name": "Low-ball knee flexion series", "focus": "Getting down to low balls",
        "setup": "Cones at knee height; feeds to the feet near the service line.",
        "dose": "5 x 10 balls", "key": "Bend the knees, not the back; eyes at ball level."},
    "movement": {
        "name": "Reaction & coverage shuttle", "focus": "First step and court coverage",
        "setup": "Coach points left/right/forward/back; player moves and shadow-swings.",
        "dose": "6 x 30 s", "key": "Split-step on the opponent's contact every time."},
    "pair_positioning": {
        "name": "Pair-as-a-unit drill (rope drill)", "focus": "Moving together with the partner",
        "setup": "Partners hold a 3 m rope while playing points from the net vs coach; they must go up and back together.",
        "dose": "4 x 3 min", "key": "Both up or both back - avoid one up / one back."},
    "middle": {
        "name": "Middle-ball communication", "focus": "Who takes the ball through the middle",
        "setup": "Feeds alternately to the middle; rule: the forehand of the right player / player diagonal to the hitter takes it, with a call.",
        "dose": "4 x 15 balls", "key": "Call 'mine'/'yours' before the bounce."},
    "rules_serve": {
        "name": "Legal serve technique", "focus": "Eliminate service faults",
        "setup": "Tape line 20 cm behind the service line, hip-height marker on a stick next to the server.",
        "dose": "50 serves", "key": "Bounce behind the line, contact at or below the waist, feet still."},
    "rules_net": {
        "name": "Net braking", "focus": "Avoid touching/crossing the net",
        "setup": "Put-away volleys from 1.5 m, stop with both feet behind a tape line 70 cm from the net.",
        "dose": "4 x 10", "key": "Brake with the outside foot before the shot."},
}


def drill_for(category: str) -> dict:
    d = DRILLS.get(category)
    return dict(d, id=category) if d else None
