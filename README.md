# GAN Feedback Loop for Content Optimization

This project is an implementation of a Generative Adversarial Network (GAN)-inspired feedback loop for content optimization using LangChain and OpenAI. It facilitates a dynamic collaboration between two language models: one responsible for generating content (the "Generator") and the other for providing constructive critique (the "Critic"). The objective is to iteratively improve the content based on both machine-generated and user-provided feedback.

## Overview

The script employs two models, a generator and a critic, that iteratively enhance the quality of a piece of content until it either meets a desired score or stagnates in quality. The main workflow follows these steps:

1. **Generator Model:** Uses the given prompt to generate initial content.
2. **Critic Model:** Evaluates the content, providing a score and detailed feedback for improvement.
3. **Iteration Process:** The generator modifies the content based on the critic's feedback, and the process repeats until a target score is reached, improvement stagnates, or a maximum number of attempts are made.

## How It Works

### Function: `gan_feedback_loop`

- **Parameters**:

  - `model_generator_llm`: The LLM instance for content generation (e.g., an OpenAI model).
  - `model_critic_llm`: The LLM instance for critiquing the generated content.
  - `prompt`: The initial content prompt to be refined.
  - `require_user_feedback`: Boolean flag indicating whether user feedback is needed during iterations.
  - `require_user_confirmation`: Boolean flag for requiring user confirmation at each step.
  - `min_score`: Minimum quality score for the content to reach before the process stops.
  - `max_attempts`: Maximum number of iterations for improving the content.
  - `stagnation_threshold`: Number of consecutive iterations without improvement before stopping.

- **Logic Flow**:

  1. The generator model generates content based on an initial prompt.
  2. The critic model evaluates the content, providing a score and suggesting improvements.
  3. The generator refines the content in response to the critique.
  4. The process iterates until a satisfactory score is reached or improvement stagnates.

### Prompt Templates

The script defines prompt templates for both models:

- **Generator Prompt**: Instructs the generator to improve clarity, engagement, logical flow, and alignment with audience needs.
- **Critic Prompt**: Instructs the critic to evaluate and provide actionable feedback in a JSON format.

### Stopping Criteria

The feedback loop terminates based on one of the following criteria:

- Desired score is reached (`min_score`).
- No significant improvement is detected after a set number of iterations (`stagnation_threshold`).
- Maximum number of attempts (`max_attempts`) is reached.
- User manually stops the process.

### Example Usage

- Load your OpenAI API key from environment variables.
- Create instances of the LLM for the generator and critic models.
- Execute the `gan_feedback_loop()` function with your desired parameters to generate and refine content.

## Requirements

- Python 3.7+
- **Libraries**:
  - `langchain_openai`
  - `langchain_core`
  - `dotenv`
  - `os`
  - `json`
  - `pprint`

## Installation

Install the necessary Python dependencies by running:

```bash
pip install langchain_openai langchain_core python-dotenv
```

## Running the Script

To run the script:

1. Ensure you have an OpenAI API key and set it in your environment variables.
2. Execute the script in a Python environment.

```bash
python gan_feedback_loop.py
```

## Usage Notes

- **User Feedback**: You can enable user feedback or manual confirmation between iterations by setting `require_user_feedback` or `require_user_confirmation` to `True`.
- **Parameters Adjustment**: You can tweak `min_score`, `max_attempts`, and `stagnation_threshold` to modify the behavior and stopping conditions of the optimization loop.

## Potential Use Cases

- **Content Writing**: Improve the quality of blog posts, articles, or marketing content.
- **Educational Content**: Optimize lecture notes, tutorials, or educational materials for clarity and engagement.
- **Business Communication**: Refine reports, proposals, or other business documents for a target audience.

## Limitations

- The model's performance depends heavily on the quality of the initial prompt and feedback mechanism.
- Stagnation detection is based solely on scoring, which may not always represent meaningful content improvement.
- Requires an OpenAI API key for LLM access.

## License

This project is licensed under the MIT License.

## Contact

If you have any questions, feel free to reach out via GitHub or open an issue.

## Acknowledgments

- Thanks to the LangChain team for providing useful resources and tools for LLM integration.

## Future Improvements

- Incorporate additional stopping criteria based on qualitative metrics.
- Integrate different models for the Generator and Critic for varied perspectives.
- Add a graphical user interface (GUI) for non-technical users to interact with the feedback loop.

---

Feel free to provide feedback or suggest further improvements!
