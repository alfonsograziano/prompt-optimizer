import os, json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from custom_prompts import prompt_optimization_job, prompt_optimization_system_prompt
from custom_prompts import prompt_critique_system_prompt, prompt_critique_request

def optimize_prompt(prompt_to_optimize, context=None):
    """
    Optimizes a given prompt by generating clarifying questions and refining it based on user input.
    
    Args:
        prompt_to_optimize (str): The prompt to optimize.
    
    Returns:
        tuple: A tuple containing the optimized prompt and a list of tuples with questions and user answers.
    """
    load_dotenv()

    # Load API key from environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")

    # Initialize the model
    model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)

    # Define JSON schema
    json_schema = {
        "title": "optimizedPrompt",
        "description": "Prompt optimized",
        "type": "object",
        "properties": {
            "optimizedPrompt": {
                "type": "string",
                "description": "A string containing the best prompt you can generate, given all the context",
            },
            "clarifyingQuestions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Questions for the user to further improve the prompt",
            },
        },
        "required": ["optimizedPrompt", "clarifyingQuestions"],
    }

    # Initialize structured LLM with schema
    structured_llm = model.with_structured_output(json_schema)

    # Create prompt template
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", prompt_optimization_system_prompt), ("user", prompt_optimization_job)]
    )

    # Combine template and model into a chain
    chain = prompt_template | structured_llm

    # Step 1: Generate clarifying questions
    response = chain.invoke({"prompt_to_optimize": prompt_to_optimize, "context": context})

    # Step 2: Collect user answers and create a list of question-answer tuples
    clarifying_questions = response["clarifyingQuestions"]
    qa_pairs = []
    for question in clarifying_questions:
        user_answer = input(f"Answer to '{question}': ")  # Collect user input
        qa_pairs.append((question, user_answer))

    # Step 3: Refine the prompt with user answers
    combined_context = (
        "\n".join(answer for _, answer in qa_pairs)  # Answers from qa_pairs as a single string
    )
    if context:  # Include additional context if provided
        combined_context += f"\n{context}"

    refined_response = chain.invoke({
        "prompt_to_optimize": prompt_to_optimize,
        "context": combined_context,  # Use the combined string for context
    })
    

    # Return the final optimized prompt and the QA pairs
    return refined_response["optimizedPrompt"], qa_pairs



def critique_prompt(prompt_to_analyze):
    """
    Critiques a given prompt and returns a JSON with reasoning and a score.

    Args:
        prompt_to_analyze (str): The prompt to analyze.

    Returns:
        dict: A JSON object containing the reasoning (text) and the score (float from 0 to 1).
    """
    load_dotenv()

    # Load API key from environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")

    # Initialize the model
    model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)

    # Define JSON schema
    json_schema = {
        "title": "promptCritique",
        "description": "Critique of a prompt including reasoning and a score.",
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "The detailed reasoning for the critique, including a breakdown of the analysis.",
            },
            "score": {
                "type": "number",
                "description": "The final score of the prompt, from 0 to 1.",
                "minimum": 0,
                "maximum": 1,
            },
        },
        "required": ["reasoning", "score"],
    }

    # Initialize structured LLM with schema
    structured_llm = model.with_structured_output(json_schema)

    # Create prompt template
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", prompt_critique_system_prompt), ("user", prompt_critique_request)]
    )

    # Combine template and model into a chain
    chain = prompt_template | structured_llm

    # Invoke the chain with the input prompt
    response = chain.invoke({"current_prompt": prompt_to_analyze})

    # Return the reasoning and score as a JSON object
    return {
        "reasoning": response["reasoning"],
        "score": response["score"]
    }

def optimize_and_benchmark(prompt):
    """
    Optimize a given prompt, critique both the original and optimized versions, 
    and compare their performance.

    Parameters:
        prompt (str): The original prompt to be optimized and critiqued.

    Returns:
        dict: A dictionary containing:
            - optimized_prompt (str): The improved version of the original prompt.
            - score_difference (float): The difference in scores between the optimized and original prompts.
            - original_critique_result_score (float): The critique score of the original prompt.
            - optimized_critique_result_score (float): The critique score of the optimized prompt.
            - original_critique_result (dict): Detailed critique results of the original prompt.
            - optimized_critique_result (dict): Detailed critique results of the optimized prompt.

    Process:
        1. Critique the original prompt using `critique_prompt`.
        2. Optimize the prompt using `optimize_prompt`.
        3. Critique the optimized prompt.
        4. Calculate the score difference.
        5. Return all relevant data for analysis and benchmarking.

    Example Usage:
        prompt = "Explain the theory of relativity in simple terms."
        results = optimize_and_benchmark(prompt)

        print("Original Prompt Critique Score:", results["original_critique_result_score"])
        print("Optimized Prompt Critique Score:", results["optimized_critique_result_score"])
        print("Score Difference:", results["score_difference"])
        print("Optimized Prompt:", results["optimized_prompt"])
    """
    original_critique_result = critique_prompt(prompt)
    optimized_prompt, qa_pairs = optimize_prompt(prompt)
    optimized_critique_result = critique_prompt(optimized_prompt)
    score_difference = optimized_critique_result["score"] - original_critique_result["score"]

    return {
        "optimized_prompt": optimized_prompt,
        "score_difference": score_difference,
        "original_critique_result_score": original_critique_result["score"],
        "optimized_critique_result_score": optimized_critique_result["score"],
        "original_critique_result": original_critique_result,
        "optimized_critique_result": optimized_critique_result
    }

if __name__ == "__main__":
    user_prompt = input(f"Add here the prompt you need to optimize: ") 

    print("Thinking about how to optimize your prompt...")
    result = optimize_and_benchmark(user_prompt)
    print("\nEvaluating the output...")
    print(json.dumps(result, indent=4))  
   