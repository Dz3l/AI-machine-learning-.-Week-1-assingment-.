# AI Tools Assignment: Mastering the AI Toolkit

This repository contains the solution for the "AI Tools Assignment". The assignment is divided into three parts:

1.  **Part 1: Theoretical Understanding**: Answers to theoretical questions about AI tools and frameworks.
2.  **Part 2: Practical Implementation**: Hands-on tasks using Scikit-learn, TensorFlow/PyTorch, and spaCy.
3.  **Part 3: Ethics & Optimization**: Discussion on ethical considerations in AI and a debugging challenge.

## Project Structure

```
.
├── README.md
├── .gitignore
├── part1_theory
│   └── theoretical_questions.md
├── part2_practical
│   ├── task1_sklearn
│   │   └── iris_classification.ipynb
│   ├── task2_tensorflow_pytorch
│   │   └── mnist_cnn.ipynb
│   └── task3_spacy
│       └── amazon_reviews_nlp.ipynb
├── part3_ethics_optimization
│   ├── ethics_and_optimization.md
│   └── buggy_script_fixed.py
├── bonus_streamlit_app         // Optional: For model deployment (skipped in this run)
│   └── app.py
└── report.pdf                  // Final consolidated report (see note below)
```

**Note on `report.pdf`**: This is a placeholder. Please see the instructions within `report.pdf` itself or the "Final Report Generation" section in the main plan on how to compile the actual report from the Markdown files and notebook outputs.

## Tools & Resources Used

*   **Frameworks**: Scikit-learn, TensorFlow, spaCy, NLTK (for VADER)
*   **Platforms**: Jupyter Notebook
*   **Languages**: Python

## Running the Code

- Ensure you have Python 3.x installed.
- Install necessary libraries:
  ```bash
  pip install scikit-learn tensorflow spacy pandas numpy matplotlib seaborn jupyterlab nltk
  python -m spacy download en_core_web_sm
  # For NLTK VADER, the notebook part2_practical/task3_spacy/amazon_reviews_nlp.ipynb handles download if needed.
  ```
- Navigate to the respective Jupyter notebooks in `part2_practical` and run the cells.
- The script `part3_ethics_optimization/buggy_script_fixed.py` can be run directly.
- The Streamlit app in `bonus_streamlit_app/app.py` was not run in this session. If you wish to run it:
  ```bash
  pip install streamlit opencv-python # Ensure opencv is installed for the app
  streamlit run bonus_streamlit_app/app.py
  # You might need to train and save model weights (mnist_model_weights.h5) from the mnist_cnn.ipynb first,
  # or let the Streamlit app train a temporary one.
  ```

## Group Members
*(Please fill in your group members' names here)*

- Member 1
- Member 2
- Member 3
- Member 4 (if applicable)
- Member 5 (if applicable)

## Presentation
A 3-minute video explaining your approach and findings will be shared on the Community platform.

## Report
The final report in PDF format (`report.pdf`) includes answers to theoretical questions, screenshots of model outputs, and ethical reflections. It is also shared as an article in the Community for peer group review. The `report.pdf` in this repository is a placeholder with instructions for its creation.

## Presentation Guidance (3-minute video)

Here are some tips for creating your 3-minute video presentation:

*   **Objective:** Clearly explain your project's approach, key implementations, and findings.
*   **All Members Participate:** Ensure every group member has a speaking role, however brief. This demonstrates teamwork.
*   **Structure Suggestions:**
    1.  **Introduction (approx. 20-30 seconds):**
        *   Briefly introduce the assignment ("AI Tools Assignment").
        *   Introduce group members.
        *   State the main tools/technologies used (Scikit-learn, TensorFlow, spaCy).
    2.  **Part 1: Theory (approx. 20-30 seconds):**
        *   Briefly mention one or two key theoretical differences or insights you found interesting (e.g., PyTorch vs TensorFlow, or spaCy's advantages). Don't try to cover all questions.
    3.  **Part 2: Practical Implementation (approx. 60-90 seconds):**
        *   **Task 1 (Scikit-learn):** Show a quick visual of the Iris classification result (e.g., accuracy).
        *   **Task 2 (TensorFlow):** Show a visual of the MNIST CNN architecture (briefly) and a key result (e.g., test accuracy >95%, sample predictions).
        *   **Task 3 (spaCy):** Show a quick example of NER and sentiment output on a sample review.
        *   Focus on *what* you did and *key outcomes*, not deep dives into code. Screen recordings of notebook outputs can be effective.
    4.  **Part 3: Ethics & Optimization (approx. 30-40 seconds):**
        *   Briefly mention one potential bias you discussed (e.g., in MNIST or Amazon reviews).
        *   Briefly mention the type of bug you fixed in the troubleshooting challenge.
    5.  **Conclusion (approx. 10-20 seconds):**
        *   Summarize key learnings or challenges.
        *   Thank the audience.
*   **Visuals:**
    *   Use slides with key points, screenshots of your code/results, or short screen recordings.
    *   Keep text on slides minimal.
*   **Pacing:**
    *   3 minutes is short! Practice your timing.
    *   Speak clearly and concisely.
*   **Tools for Video Creation:** Use any familiar tools like Zoom (record meeting), OBS Studio, PowerPoint recording, or online video editors.
*   **Submission:** Share the video on the Community platform as instructed.
