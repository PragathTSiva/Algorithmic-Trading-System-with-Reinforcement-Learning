# README.md
# Team Bios
## Jadyn Chowdhury, <i>Team Lead</i>
<p>
<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTLrW0cw63kvIuiAQ0ctuy3bX5qffd8s4APEw&s" alt="Me" width="150" style="float: left; margin-right: 15px; border-radius: 50%;">

Hi! My name is Jadyn Chowdhury. I am currently pursuing a dual degree in Computer Science and Aerospace Engineering at the University of Illinois at Urbana-Champaign, with an expected graduation date of May 2025. I am actively seeking full-time opportunities beginning after graduation, where I can apply my technical skills and interdisciplinary background to impactful challenges.

My project experience spans machine learning, autonomous systems, and full-stack development. I recently developed a 2D Relative-Attentional Transformer for portfolio optimization, designed to process historical stock data while accounting for liquidity constraints and market impact costs. I’ve also built an Android weather application that uses the Google Gemini API to provide intelligent weather insights. Additionally, I created a multi-drone synchronization system using motion capture and custom control logic, as well as an autonomous robotic drawing pipeline with a UR3 robotic arm and ROS, combining OpenCV-based image processing and inverse kinematics for path planning.

Professionally, I interned at DataSoft Systems as a Full-Stack Software Engineer, where I developed a CRM platform with integrated machine learning models for analytics and forecasting. I also led the development of a Django-based electronic health record system for Taiba Medical Centre, deploying scalable infrastructure via Terraform and AWS. 

My academic background includes coursework in Machine Learning, Deep Learning for Computer Vision, Artificial Intelligence, Text Information Systems, Distributed Systems and Stochastic Processes, all of which have reinforced and complemented my hands-on work.

I’m particularly interested in roles at the intersection of software, AI/ML, and quantitative systems in research, engineering, or financial technology and I’m excited to keep building toward long-term contributions in these spaces.

🌐 Learn more about me and my portfolio: [jadyn-chowdhury.me](https://jadyn-chowdhury.me/)

💼 Access my professional background and resume: [LinkedIn](https://www.linkedin.com/in/jadyn-chowdhury)

📫 Reach out: [jadynchowdhury123@gmail.com](jadynchowdhury123@gmail.com)
</p>

## Nico Luo, <i>Team Member</i>
<p>
<img src="https://media.licdn.com/dms/image/v2/C4D03AQFlvLqfLXOZcw/profile-displayphoto-shrink_400_400/profile-displayphoto-shrink_400_400/0/1647652801836?e=1750896000&v=beta&t=BN8NgU68o2dL6SaYONoTU0pkg5T3FSfY50vrh3UcPic" alt="Me" width="150" style="float: left; margin-right: 15px; border-radius: 50%;">

Hello! My name is Nico Luo and I am a senior studying Computer Science at the University of Illinois at Urbana-Champaign. I will be pursuing a master's degree in Computer Science with a focus on ML/AI, and I will graduate in May 2026. I am open to work as soon as I graduate (Summer 2026) and would be grateful for any and all opportunities to apply my skills and learn about the real-world problems that exist today.

During this upcoming Summer 2025, I will be working at Cadence Design Systems as a Software R&D intern, specifically working on machine learning tools for hardware verification. During the previous summer, I interned at a small startup called Pure Math AI, where I performed many data science tasks, including creating an AWS pipeline to process poorly structured XML data and writing Python scripts for efficient data conversion and processing. I also helped develop a natural language to SQL query tool using LLMs/RAG, during which I applied my machine learning engineering skills.

On my own time, I am currently working on a sports betting tool that will provide users with NBA player prop predictions by developing a custom machine learning model. This model will be trained on a multitude of data points gathered from many sources that include player data, game data, sportsbook data, and much more. This tool will also include a sleek and easy-to-use interface where users can quickly find any information to assist in making the best bets.

In addition to my industry experience and completed/ongoing projects, I have also taken many relevant courses during my time at UIUC. These include Machine Learning, Artificial Intelligence, Database Systems, Computer Security, Communication Networks, and many more.

I am passionate about ML/AI, computer vision, and software engineering and would be interested in roles related to these fields. Please reach out to me if you have any questions or would like to chat! 

Check out my LinkedIn: [https://www.linkedin.com/in/nicoluo/](https://www.linkedin.com/in/nicoluo/)

Contact me @ [nicoluo@gmail.com](nicoluo@gmail.com)


## Pragath Siva, <i>Team Member</i>
<p>
<img src="img/PragathPic.png" alt="Pragath Siva" width="150" style="float: left; margin-right: 15px; border-radius: 50%;">

Hi! My name is Pragath Siva. I am majoring in Computer Science and Economics at the University of Illinois at Urbana-Champaign, with an expected graduation date of December 2025. I am actively seeking software-engineering, quantitative-developer, or ML/AI research-scientist roles for Spring 2026 and Summer 2026, with a strong preference for work that pushes the state of the art in large-language-model systems.

My project experience spans AI developer tooling, blockchain protocols, browser automation, and applied machine learning. I recently won an Aptos bounty by delivering an AI Dev Assistant that fuses retrieval-augmented generation with structured documentation pipelines to return verifiable code snippets. I also engineered an on-chain IPO flow for memecoin launches, combining Solidity with encrypted weighted-price clearing. Most recently, I built Sophia Automata—a dual-agent research platform that orchestrates browser sessions, semantically ranks web sources, and injects structured summaries into Google Docs in under thirty seconds, demonstrating autonomous LLM agents coordinating across heterogeneous tasks.

My academic background includes advanced coursework in algorithms, computer architecture, statistics, applied econometrics, game theory, and high-frequency trading technology. Taken together with my project work, these studies fuel a passion for  ML/AI, computer vision and financial technology.

💼 LinkedIn: [https://www.linkedin.com/in/pragathsiva](https://www.linkedin.com/in/pragathsiva)  
🔗 GitHub: [https://github.com/PragathTSiva](https://github.com/PragathTSiva)  
📫 Reach out: [ptsiva2@illinois.edu](mailto:ptsiva2@illinois.edu)
</p>


```text
agitrader/
├── main.ipynb              # Central Colab notebook for training, logging, evaluation
├── README.md               # Project overview, structure, usage instructions
├── requirements.txt        # pip dependencies (e.g., gym, sb3, pandas, etc.)
├── .gitignore              # Files to exclude from Git version control
│
├── data/                   # Data loading, parsing, and selection tools
│   ├── raw/                # (Optional) Unprocessed IEX DEEP/TOPS data files
│   │   └── iex_aapl_2021-03-15.csv
│   ├── parsed/             # Downsampled LOB data used in training
│   │   └── lob_day1_aapl.csv
│   ├── preprocess.py       # IEX DEEP parser and top-5 LOB downsampler
│   └── selector.py         # Selects eventful/volatile days for training
│
├── env/                    # Gym-compatible trading environment
│   ├── __init__.py
│   ├── lob_env.py          # Main environment class (reset, step, space definitions)
│   ├── state_builder.py    # Constructs LOB observation feature vectors
│   └── fills.py            # Simulates order fills, queueing, and latency
│
├── agent/                  # PPO + LSTM agent logic and training
│   ├── __init__.py
│   ├── policy.py           # Optional custom LSTM or Transformer policy
│   ├── train.py            # PPO training loop and agent initialization
│   └── callbacks.py        # Logging, checkpointing, and evaluation hooks
│
├── reward/                 # Reward signal components and shaping logic
│   ├── __init__.py
│   └── reward.py           # Implements spread capture, drawdown, inventory penalties
│
├── evaluation/             # Post-training strategy discovery and visualization
│   ├── __init__.py
│   ├── logger.py           # Logs episode data: actions, prices, rewards, etc.
│   ├── cluster.py          # Performs t-SNE/PCA and strategy clustering
│   └── plot.py             # Visualizations of PnL, actions, and behavioural clusters
│
└── config/                 # Global settings and experiment constants
    ├── __init__.py
    └── settings.py         # Defines sampling rate, asset symbol, sequence length, etc.
