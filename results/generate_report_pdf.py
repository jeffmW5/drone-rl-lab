"""Generate LEARNING_REPORT.pdf from content."""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, white, black
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether, HRFlowable
)

# Colors
DARK_BLUE = HexColor("#1a365d")
MED_BLUE = HexColor("#2b6cb0")
LIGHT_BLUE = HexColor("#ebf4ff")
LIGHT_GRAY = HexColor("#f7fafc")
MED_GRAY = HexColor("#e2e8f0")
DARK_GRAY = HexColor("#2d3748")
CODE_BG = HexColor("#edf2f7")
ACCENT = HexColor("#dd6b20")  # Orange accent

def build_pdf():
    output_path = "LEARNING_REPORT.pdf"
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch,
        leftMargin=0.85*inch,
        rightMargin=0.85*inch,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    styles.add(ParagraphStyle(
        "MainTitle", parent=styles["Title"],
        fontSize=28, leading=34, textColor=DARK_BLUE,
        spaceAfter=6, alignment=TA_CENTER,
        fontName="Helvetica-Bold",
    ))
    styles.add(ParagraphStyle(
        "Subtitle", parent=styles["Normal"],
        fontSize=13, leading=18, textColor=MED_BLUE,
        spaceAfter=30, alignment=TA_CENTER,
        fontName="Helvetica",
    ))
    styles.add(ParagraphStyle(
        "SectionHead", parent=styles["Heading1"],
        fontSize=18, leading=24, textColor=DARK_BLUE,
        spaceBefore=24, spaceAfter=10,
        fontName="Helvetica-Bold",
        borderWidth=0, borderColor=MED_BLUE,
        borderPadding=0,
    ))
    styles.add(ParagraphStyle(
        "SubHead", parent=styles["Heading2"],
        fontSize=13, leading=17, textColor=MED_BLUE,
        spaceBefore=14, spaceAfter=6,
        fontName="Helvetica-Bold",
    ))
    styles.add(ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=10, leading=14.5, textColor=DARK_GRAY,
        spaceAfter=8, fontName="Helvetica",
    ))
    styles.add(ParagraphStyle(
        "BodyBold", parent=styles["Normal"],
        fontSize=10, leading=14.5, textColor=DARK_GRAY,
        spaceAfter=8, fontName="Helvetica-Bold",
    ))
    styles.add(ParagraphStyle(
        "BulletItem", parent=styles["Normal"],
        fontSize=10, leading=14.5, textColor=DARK_GRAY,
        spaceAfter=4, fontName="Helvetica",
        leftIndent=20, bulletIndent=8,
    ))
    styles.add(ParagraphStyle(
        "CodeBlock", parent=styles["Normal"],
        fontSize=9, leading=13, textColor=HexColor("#1a202c"),
        fontName="Courier", backColor=CODE_BG,
        leftIndent=12, rightIndent=12,
        spaceBefore=4, spaceAfter=4,
        borderWidth=0.5, borderColor=MED_GRAY, borderPadding=6,
    ))
    styles.add(ParagraphStyle(
        "Insight", parent=styles["Normal"],
        fontSize=10, leading=14.5, textColor=HexColor("#744210"),
        spaceAfter=10, fontName="Helvetica-Oblique",
        leftIndent=16, rightIndent=16,
        backColor=HexColor("#fefcbf"), borderPadding=8,
        borderWidth=0.5, borderColor=HexColor("#ecc94b"),
    ))
    styles.add(ParagraphStyle(
        "TableCell", parent=styles["Normal"],
        fontSize=8.5, leading=11, textColor=DARK_GRAY,
        fontName="Helvetica",
    ))
    styles.add(ParagraphStyle(
        "TableHeader", parent=styles["Normal"],
        fontSize=8.5, leading=11, textColor=white,
        fontName="Helvetica-Bold",
    ))

    story = []

    # ── TITLE PAGE ──
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("Drone RL Lab", styles["MainTitle"]))
    story.append(Paragraph("Learning Report", styles["MainTitle"]))
    story.append(Spacer(1, 0.3*inch))
    story.append(HRFlowable(width="40%", thickness=2, color=ACCENT, spaceAfter=12))
    story.append(Paragraph(
        "How our reinforcement learning drone racing system works,<br/>"
        "what every variable means, and where we stand vs the competition.",
        styles["Subtitle"]
    ))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("Target: Sub-5s on Kaggle Level 2 (Top 3)", ParagraphStyle(
        "Target", parent=styles["Normal"], fontSize=14, textColor=ACCENT,
        alignment=TA_CENTER, fontName="Helvetica-Bold",
    )))
    story.append(PageBreak())

    def make_table(headers, rows, col_widths=None):
        """Build a styled table."""
        header_cells = [Paragraph(h, styles["TableHeader"]) for h in headers]
        data = [header_cells]
        for row in rows:
            data.append([Paragraph(str(c), styles["TableCell"]) for c in row])

        t = Table(data, colWidths=col_widths, repeatRows=1)
        style_cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), MED_BLUE),
            ("TEXTCOLOR", (0, 0), (-1, 0), white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("TOPPADDING", (0, 0), (-1, 0), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, MED_GRAY),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 1), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
        ]
        # Alternating row colors
        for i in range(1, len(data)):
            if i % 2 == 0:
                style_cmds.append(("BACKGROUND", (0, i), (-1, i), LIGHT_GRAY))
        t.setStyle(TableStyle(style_cmds))
        return t

    def section(num, title):
        story.append(Paragraph(f"{num}. {title}", styles["SectionHead"]))
        story.append(HRFlowable(width="100%", thickness=1, color=MED_BLUE, spaceAfter=10))

    def sub(title):
        story.append(Paragraph(title, styles["SubHead"]))

    def body(text):
        story.append(Paragraph(text, styles["Body"]))

    def bold(text):
        story.append(Paragraph(text, styles["BodyBold"]))

    def bullet(text):
        story.append(Paragraph(f"\u2022  {text}", styles["BulletItem"]))

    def code(text):
        story.append(Paragraph(text.replace("\n", "<br/>"), styles["CodeBlock"]))

    def insight(text):
        story.append(Paragraph(f"\u26a0  {text}", styles["Insight"]))

    def gap():
        story.append(Spacer(1, 8))

    # ── SECTION 1: HOW PPO WORKS ──
    section("1", "How PPO Works (Plain English)")
    bold("Reinforcement Learning = an agent learns by trial and error.")
    body("It takes actions in an environment, gets rewards, and updates its strategy to get more reward over time.")
    gap()
    bold("PPO (Proximal Policy Optimization) is our specific RL algorithm. Here's the loop:")
    code(
        "1. Fly the drone (take actions) for 8 steps across 64 parallel simulations<br/>"
        "2. Collect: what did I see? what did I do? what reward did I get?<br/>"
        "3. Compute 'advantages' -- was each action better or worse than expected?<br/>"
        "4. Update the neural network weights to make good actions more likely<br/>"
        "5. Repeat from step 1"
    )
    gap()
    body("The 'proximal' part means PPO is careful -- it clips how much the policy can change per update "
         "(controlled by <b>clip_coef = 0.26</b>). This prevents the agent from making a big change that "
         "accidentally destroys what it already learned.")
    gap()
    bold("Two networks work together:")
    bullet("<b>Actor</b> (policy): 'Given what I see, what should I do?' -- outputs action probabilities")
    bullet("<b>Critic</b> (value): 'Given what I see, how much total reward do I expect?' -- outputs a single number")
    gap()
    bold("GAE (Generalized Advantage Estimation):")
    body("Computes 'how much better was this specific action compared to what I normally do in this state?' "
         "Positive advantage = 'do more of this', negative = 'do less.' This is what actually drives learning.")

    # ── SECTION 2: NEURAL NETWORK ──
    section("2", "The Neural Network")
    code(
        "Input (73 dims) --> [Linear 64] --> [Tanh] --> [Linear 64] --> [Tanh] --> Output<br/>"
        "                    (both actor and critic have this structure)"
    )
    gap()
    sub("Actor Head")
    body("Outputs mean of 4 actions + learned log-standard-deviation.")
    bullet("Log-std initialized to <b>[-1, -1, -1, 1]</b>")
    bullet("This means: low exploration on roll/pitch/yaw (already bounded), high exploration on thrust (needs more experimenting)")
    bullet("Final layer uses Tanh to bound outputs to [-1, 1]")
    gap()
    sub("Critic Head")
    body("Outputs a single value estimate (expected future reward).")
    gap()
    sub("Size and Initialization")
    body("<b>~9,000 trainable parameters.</b> This is tiny by modern standards -- the network is small because "
         "the observation space is small and the physics are relatively low-dimensional.")
    body("<b>Orthogonal initialization</b> helps with gradient flow in Tanh networks. Actor final layer uses "
         "std=0.01 (start with near-zero actions = hover), critic final layer uses std=1.0.")

    # ── SECTION 3: OBSERVATIONS ──
    section("3", "What the Agent Sees (73 Observation Dimensions)")
    body("Every simulation step, the agent receives a vector of 73 numbers:")
    gap()
    story.append(make_table(
        ["Component", "Dims", "Description"],
        [
            ["pos", "3", "Drone position [x, y, z] in meters"],
            ["quat", "4", "Drone orientation as quaternion [x, y, z, w]"],
            ["vel", "3", "Linear velocity [vx, vy, vz] m/s"],
            ["ang_vel", "3", "Angular velocity [wx, wy, wz] rad/s"],
            ["local_samples", "30", "Relative positions to next 10 trajectory waypoints (3 coords each)"],
            ["prev_obs frame 1", "13", "Previous step: [pos, quat, vel, ang_vel]"],
            ["prev_obs frame 2", "13", "Two steps ago: [pos, quat, vel, ang_vel]"],
            ["last_action", "4", "What the agent did last step [roll, pitch, yaw, thrust]"],
            ["Total", "73", ""],
        ],
        col_widths=[1.4*inch, 0.6*inch, 4.2*inch],
    ))
    gap()
    insight("The raw environment provides gates_pos, gates_quat, obstacles_pos, etc. But the RandTrajEnv "
            "wrapper consumes gate positions to compute a trajectory, then gives the agent local_samples "
            "(relative positions to the next 10 waypoints). The agent sees 'where should I fly next' rather "
            "than 'where are the gates.'")
    body("<b>With n_obs=0:</b> No prev_obs, total drops to 47 dims. Faster to train but agent can't infer "
         "acceleration from position history.")

    # ── SECTION 4: ACTIONS ──
    section("4", "What the Agent Does (4 Action Dimensions)")
    story.append(make_table(
        ["Action", "Range", "What it controls"],
        [
            ["Roll", "[-1, 1]", "Tilt left/right (bank for turning)"],
            ["Pitch", "[-1, 1]", "Tilt forward/back (controls forward speed)"],
            ["Yaw", "[-1, 1]", "Rotate around vertical axis (blocked to 0 by wrapper)"],
            ["Thrust", "[-1, 1]", "Vertical force (up/down, also affected by tilt)"],
        ],
        col_widths=[1.2*inch, 1*inch, 4*inch],
    ))
    gap()
    body("The <b>NormalizeActions</b> wrapper maps [-1, 1] to the drone's actual control ranges. "
         "The <b>AngleReward</b> wrapper blocks yaw to 0 (the drone doesn't need to rotate, just bank and pitch).")

    # ── SECTION 5: REWARD FUNCTION ──
    section("5", "The Reward Function")
    body("Every step, the agent gets a reward signal that tells it how well it's doing:")
    code("total_reward = base_reward + angle_penalty + energy_penalty + smoothness_penalties")

    sub("Base Reward: Follow the Trajectory")
    code("reward = exp(-2.0 * distance_to_next_waypoint)")
    story.append(make_table(
        ["Distance", "Reward", "Interpretation"],
        [
            ["0 m", "1.0", "Perfect -- right on the trajectory"],
            ["0.5 m", "0.37", "Close but not great"],
            ["1.0 m", "0.14", "Drifting off course"],
            ["2.0 m", "0.02", "Basically zero reward"],
            ["Crashed", "-1.0", "Heavy penalty"],
        ],
        col_widths=[1.2*inch, 1*inch, 4*inch],
    ))
    gap()
    body("This exponential shape means the agent gets much more reward for being very close vs. roughly close. "
         "It creates a strong gradient toward the trajectory.")

    sub("Penalties (Subtract from Base Reward)")
    story.append(make_table(
        ["Penalty", "Formula", "Weight", "Purpose"],
        [
            ["Angle", "-rpy_coef * norm(euler_angles)", "0.06", "Don't tilt excessively"],
            ["Energy", "-act_coef * thrust^2", "0.02", "Don't waste energy"],
            ["Thrust smoothness", "-d_act_th_coef * (delta thrust)^2", "0.4", "Don't jerk the throttle"],
            ["Roll/pitch smoothness", "-d_act_xy_coef * sum(delta roll/pitch)^2", "1.0", "Don't jerk the attitude"],
        ],
        col_widths=[1.4*inch, 2.2*inch, 0.6*inch, 2*inch],
    ))
    gap()
    insight("The reward is about trajectory-following, NOT about passing through gates. The trajectory happens "
            "to pass through gates, but the agent only knows 'follow these waypoints.' This is why Level 2 "
            "(randomized gates) breaks things -- the trajectory might not go through the moved gates.")

    # ── SECTION 6: HYPERPARAMETERS ──
    section("6", "Every Hyperparameter Explained")

    sub("PPO Algorithm Knobs")
    story.append(make_table(
        ["Parameter", "Value", "What It Does", "Turn Up", "Turn Down"],
        [
            ["learning_rate", "0.0015", "Size of weight updates", "Faster, risk oscillation", "Slower, more stable"],
            ["gamma", "0.94", "Discount for future rewards", "Values future more", "Focuses on immediate"],
            ["gae_lambda", "0.97", "Advantage smoothing", "Lower bias, higher variance", "Higher bias, more stable"],
            ["clip_coef", "0.26", "Max policy change", "Bigger strategy shifts", "Smaller, safer updates"],
            ["ent_coef", "0.007", "Exploration bonus", "More random exploration", "More exploitation"],
            ["vf_coef", "0.7", "Value loss weight", "Better value predictions", "Focus on policy"],
            ["max_grad_norm", "1.5", "Gradient clipping", "Larger param changes", "Prevents instability"],
            ["update_epochs", "10", "Reuse batch N times", "More learning per batch", "Less overfitting"],
            ["num_minibatches", "8", "Split batch into N", "More small updates", "Fewer large updates"],
        ],
        col_widths=[1.1*inch, 0.5*inch, 1.3*inch, 1.5*inch, 1.5*inch],
    ))

    gap()
    sub("Scale Knobs")
    story.append(make_table(
        ["Parameter", "CPU", "GPU", "What It Does"],
        [
            ["num_envs", "64", "1024", "Parallel simulations. More = smoother gradients + faster"],
            ["num_steps", "8", "8", "Rollout length before update"],
            ["total_timesteps", "500k", "3-10M", "Total training budget"],
            ["n_obs", "0 or 2", "2", "Observation history frames"],
            ["budget_seconds", "600", "1800-7200", "Wall-clock training cutoff"],
        ],
        col_widths=[1.2*inch, 0.7*inch, 0.8*inch, 3.5*inch],
    ))
    gap()
    bold("Batch size = num_envs x num_steps")
    bullet("CPU: 64 x 8 = <b>512</b> samples per PPO update")
    bullet("GPU: 1024 x 8 = <b>8,192</b> samples per PPO update (16x more signal)")

    gap()
    sub("Reward Shaping Knobs")
    story.append(make_table(
        ["Parameter", "Value", "Effect of Increasing"],
        [
            ["rpy_coef", "0.06", "Drone stays more level (less aggressive banking)"],
            ["act_coef", "0.02", "Uses less thrust (more efficient but potentially slower)"],
            ["d_act_th_coef", "0.4", "Smoother thrust changes (less jerky altitude control)"],
            ["d_act_xy_coef", "1.0", "Smoother roll/pitch changes (less aggressive maneuvering)"],
        ],
        col_widths=[1.2*inch, 0.7*inch, 4.3*inch],
    ))
    gap()
    insight("Higher smoothness penalties = safer, smoother flight but potentially slower lap times. "
            "The Kaggle winners likely tuned these down to allow more aggressive flying.")

    # ── SECTION 7: OUR EXPERIMENTS ──
    section("7", "Our Experiments So Far")

    sub("Experiment 010 -- Racing Baseline (Level 0, CPU)")
    bullet("<b>Config:</b> n_obs=0, 500k steps, 64 envs, 600s budget")
    bullet("<b>Result:</b> reward = 7.36, completed all 500k steps")
    bullet("<b>Sim test:</b> 5/5 finishes, average 13.36s (within 0.024s of reference model)")
    bullet("<b>Lesson:</b> The pipeline works. With n_obs=0, the agent converges in 10 min on CPU.")

    gap()
    sub("Experiment 013 -- n_obs Fix (Level 0, CPU)")
    bullet("<b>Config:</b> n_obs=2, 500k steps, 64 envs, 600s budget")
    bullet("<b>Result:</b> reward = 5.02, only completed 297k/500k steps (hit time limit)")
    bullet("<b>Sim test:</b> 0/5 finishes (crashes on every run)")
    bullet("<b>Lesson:</b> n_obs=2 makes observations 55% larger (47 to 73 dims). Severely undertrained on CPU. Needs GPU.")

    gap()
    sub("Level 2 Benchmark -- All Controllers")
    body("Everyone tested on the competition level (randomized gates + physics):")
    story.append(make_table(
        ["Controller", "Type", "Avg Time", "Finishes", "Gates"],
        [
            ["State controller", "Trajectory following", "5.96s", "1/5", "0-4"],
            ["PID attitude", "Classical PID", "8.59s", "2/5", "0-4"],
            ["Their RL (pre-trained)", "RL (trained on L0)", "7.25s", "0/5", "0-3"],
            ["Our RL (exp_013)", "RL (undertrained)", "4.30s", "0/5", "0-2"],
        ],
        col_widths=[1.5*inch, 1.3*inch, 0.9*inch, 0.8*inch, 0.7*inch],
    ))
    gap()
    insight("Nobody reliably finishes Level 2. All controllers follow trajectories built from fixed "
            "waypoints that don't adapt to randomized gate positions.")

    # ── SECTION 8: KAGGLE LEADERBOARD ──
    section("8", "Kaggle Competition Leaderboard")
    body("<b>Competition:</b> lsy-drone-racing-ws-25 (TUM, Winter Semester 2025)")
    body("<b>Task:</b> Complete a drone racing course on Level 2 (randomized gates + physics)")
    body("<b>Scoring:</b> Average lap time across evaluation runs")
    gap()

    # Highlight top 3
    kaggle_data = [
        ["1", "Team Y", "3.39", "7"],
        ["2", "Group6", "4.89", "17"],
        ["3", "Limo", "5.02", "17"],
        ["4", "Liangyu Chen, Tuo Yang", "5.61", "9"],
        ["5", "Jai Seth", "9.56", "10"],
        ["6", "Elena Kuznetsova", "22.51", "1"],
        ["7", "RandomUsername2374", "24.29", "1"],
        ["8", "Marcel Rath", "27.07", "1"],
        ["9", "Radu Cristian", "28.19", "1"],
        ["10", "Yufei Hua", "29.99", "1"],
    ]
    header = [Paragraph(h, styles["TableHeader"]) for h in ["Rank", "Team", "Score (s)", "Submissions"]]
    tdata = [header]
    for row in kaggle_data:
        tdata.append([Paragraph(c, styles["TableCell"]) for c in row])

    kt = Table(tdata, colWidths=[0.6*inch, 2.5*inch, 1*inch, 1*inch], repeatRows=1)
    kcmds = [
        ("BACKGROUND", (0, 0), (-1, 0), MED_BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("GRID", (0, 0), (-1, -1), 0.5, MED_GRAY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 1), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
        # Highlight top 3
        ("BACKGROUND", (0, 1), (-1, 1), HexColor("#fefce8")),  # Gold
        ("BACKGROUND", (0, 2), (-1, 2), HexColor("#f0f4ff")),  # Silver
        ("BACKGROUND", (0, 3), (-1, 3), HexColor("#fff5eb")),  # Bronze
    ]
    for i in range(4, len(tdata)):
        if i % 2 == 0:
            kcmds.append(("BACKGROUND", (0, i), (-1, i), LIGHT_GRAY))
    kt.setStyle(TableStyle(kcmds))
    story.append(kt)
    gap()
    story.append(Paragraph(
        "Our target: sub-5s (top 3)",
        ParagraphStyle("TargetInline", parent=styles["Body"], fontSize=12,
                       textColor=ACCENT, fontName="Helvetica-Bold")
    ))
    body("The top teams submitted 7-17 times, suggesting significant iteration. "
         "The bottom teams submitted once (likely the default controllers).")

    # ── SECTION 9: THE GAP ──
    section("9", "The Gap: Where We Are vs Where We Need to Be")

    sub("Current Status")
    bullet("Our best model (exp_010) completes Level 0 in 13.36s but can't handle Level 2")
    bullet("Level 2 breaks ALL existing controllers because trajectories don't adapt to moved gates")
    bullet("We've only trained on CPU with 64 envs and 500k steps")

    sub("What the Kaggle Winners Likely Did")
    bullet("<b>Trained directly on Level 2</b> -- agent experiences randomized gates during training")
    bullet("<b>Used GPU + many envs</b> -- 1024+ parallel simulations")
    bullet("<b>Trained for millions of steps</b> -- 3-10M+ timesteps")
    bullet("<b>Tuned reward coefficients</b> -- reduced smoothness penalties for faster, more aggressive flight")
    bullet("<b>Possibly used curriculum learning</b> -- start easy, progressively add difficulty")

    sub("Our GPU Plan")
    story.append(make_table(
        ["Experiment", "Level", "Steps", "Envs", "Purpose"],
        [
            ["exp_014", "0", "1.5M", "1024", "Validate GPU, converge n_obs=2"],
            ["exp_015", "2", "3M", "1024", "First competition attempt"],
            ["exp_016", "2", "10M", "1024", "Extended if 3M isn't enough"],
        ],
        col_widths=[0.9*inch, 0.5*inch, 0.7*inch, 0.6*inch, 3.5*inch],
    ))
    gap()
    sub("Why GPU Changes Everything")
    bullet("1024 envs x 8 steps = <b>8,192 samples per update</b> (vs 512 on CPU)")
    bullet("GPU parallelizes the matrix math across all envs simultaneously")
    bullet("Estimated: 3M steps in ~5-10 minutes on RTX 3090 (vs hours on CPU)")
    bullet("n_obs=2 won't be a bottleneck -- GPU handles the larger obs easily")

    # ── SECTION 10: GLOSSARY ──
    section("10", "Glossary")
    story.append(make_table(
        ["Term", "Definition"],
        [
            ["PPO", "Proximal Policy Optimization -- the RL algorithm"],
            ["Actor", "Neural network that decides actions"],
            ["Critic", "Neural network that estimates expected future reward"],
            ["GAE", "Generalized Advantage Estimation -- measures how good an action was"],
            ["Rollout", "A sequence of steps collected before updating the network"],
            ["Batch size", "num_envs * num_steps = total samples per PPO update"],
            ["Epoch", "One pass through the entire batch during PPO update"],
            ["Entropy", "Measure of randomness in the policy (higher = more exploration)"],
            ["Clipping", "Limiting how much the policy can change per update"],
            ["Advantage", "How much better an action was vs the average for that state"],
            ["Discount (gamma)", "How much to value future vs immediate rewards"],
            ["Obs stacking", "Including past observations so the agent can see trends"],
            ["Level 0", "Perfect knowledge -- fixed gates, no randomization"],
            ["Level 1", "Randomized physics (mass, inertia) but fixed gates"],
            ["Level 2", "Competition level -- randomized gates + physics + obstacles"],
        ],
        col_widths=[1.3*inch, 4.9*inch],
    ))

    # ── BUILD ──
    def add_page_number(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(HexColor("#a0aec0"))
        canvas.drawRightString(7.5*inch, 0.5*inch, f"Page {doc.page}")
        canvas.drawString(0.85*inch, 0.5*inch, "Drone RL Lab -- Learning Report")
        canvas.restoreState()

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f"PDF generated: {output_path}")


if __name__ == "__main__":
    build_pdf()
