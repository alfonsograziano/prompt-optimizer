from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os, json, pprint

load_dotenv()

def gan_feedback_loop(
    model_generator_llm,  # LLM instance for Model 1
    model_critic_llm,     # LLM instance for Model 2
    prompt,
    require_user_feedback=False,
    require_user_confirmation=False,
    min_score=8,
    max_attempts=5,
    stagnation_threshold=2
):
    """
    Implements a Generative-Adversarial inspired feedback loop between two LLMs using LangChain.
    Now includes a chain of thought approach for improved reasoning.
    
    Parameters:
    - model_generator_llm: The LLM instance for Model 1 (e.g., OpenAI model).
    - model_critic_llm: The LLM instance for Model 2.
    - prompt: The initial prompt to generate content.
    - require_user_feedback: Boolean flag for user feedback between iterations.
    - require_user_confirmation: Boolean flag for user confirmation between iterations.
    - min_score: The minimum score to reach before stopping.
    - max_attempts: Maximum number of iterations.
    - stagnation_threshold: Number of iterations with no improvement before stopping.
    
    Returns:
    - A dictionary containing the final results, including the reason for stopping, final score, 
      generated content, critique history, and any user feedback incorporated.
    """
    # Define PromptTemplates
    # Model 1 PromptTemplate - Chain of Thought Generation
    prompt_optimization_system_prompt = "You are an expert content generator. Think step-by-step to provide a chain of thought to derive a high-quality output."
    prompt_optimization_job = """
Your task is to generate high-quality content following a chain of thought. Start by breaking down the problem step-by-step, detailing your reasoning before providing the final content. Take into account the critique if available and use that to iterate on the chain of thoughts so that you can improve your answer.

Prompt:
{initial_prompt}

Chain of Thought:
{chain_of_thought}
Critique:
{critique}

If there are any clarifying questions from the critique, answer them as part of your chain of thought.
Clarifying Questions:
{clarifying_questions}
"""
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_optimization_system_prompt),
            ("user", prompt_optimization_job)
        ]
    )

    model1_chain = prompt_template | model_generator_llm
    
    # Model 2 PromptTemplate - Reasoning Critique
    model2_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert critic specializing in evaluating reasoning chains. Your goal is to provide a comprehensive critique of the chain of thought and identify any flaws in the reasoning. Provide your response in a structured JSON format, and consider the history of critiques and scores to help identify improvements or recurring issues."),
            ("user", "Chain of Thought: {chain_of_thought}\nCritique History: {critique_history}\nScore History: {score_history}")
        ]
    )
    model2_chain = model2_prompt_template | model_critic_llm

    # Initialization
    critique_history = []
    score_history = []
    user_feedback_incorporated = []
    reason_to_stop = ""
    final_score = 0
    attempts = 0
    stagnation_counter = 0
    previous_score = 0
    user_feedback = ""
    
    # Main loop
    while attempts < max_attempts:
        attempts += 1
        
        # Step 1: Model 1 generates chain of thought and content
        if attempts == 1:
           model1_inputs = {
                "initial_prompt": prompt,
                "chain_of_thought": "",
                "critique": "",
                "clarifying_questions": ""
            }
           generated_response = model1_chain.invoke(model1_inputs)
        else:
            generated_response = model1_chain.invoke({
                "initial_prompt": prompt,
                "chain_of_thought": chain_of_thought,
                "critique": critique_json.get('Critique', ''),
                "clarifying_questions": "\n".join(critique_json.get('ClarifyingQuestions', []))
            })
        
        chain_of_thought = generated_response.content

        if require_user_confirmation:
            print("\nGenerated Chain of Thought:")
            print(chain_of_thought)
            user_input = input("\nPress Enter to continue or type 'stop' to terminate: ")
            if user_input.lower() == 'stop':
                reason_to_stop = "User terminated the process."
                break
        
        # Step 2: Model 2 critiques the chain of thought
        model2_inputs = {
            "chain_of_thought": chain_of_thought,
            "critique_history": json.dumps(critique_history),
            "score_history": json.dumps(score_history)
        }
        critique_json = model2_chain.invoke(model2_inputs)

        critique_history.append(critique_json)
        score_history.append(critique_json.get('ReasoningScore', 0))
        
        final_score = critique_json.get('ReasoningScore', 0)
        
        # Check for stagnation
        if final_score <= previous_score:
            stagnation_counter += 1
        else:
            stagnation_counter = 0  # Reset if improvement is detected
        previous_score = final_score
        
        # Exit conditions
        if final_score >= min_score:
            reason_to_stop = "Desired score reached."
            break
        if stagnation_counter >= stagnation_threshold:
            reason_to_stop = "No significant improvement detected."
            break
        
        # Step 3: User feedback if required
        if critique_json.get('ClarifyingQuestions'):
            if require_user_feedback:
                print("\nModel 2 has the following questions:")
                for question in critique_json['ClarifyingQuestions']:
                    print(f"- {question}")
                user_feedback = input("\nPlease provide answers to the above questions: ")
                user_feedback_incorporated.append(user_feedback)
            else:
                # Append the clarifying questions for the next generator iteration
                clarifying_questions = "\n".join(critique_json['ClarifyingQuestions'])
                user_feedback_incorporated.append(clarifying_questions)
        else:
            user_feedback = ""
        
        # Optional user confirmation
        if require_user_confirmation:
            print("\nCritique:")
            print(critique_json.get('Critique', ''))
            print(f"Score: {final_score}")
            user_input = input("\nPress Enter to continue or type 'stop' to terminate: ")
            if user_input.lower() == 'stop':
                reason_to_stop = "User terminated the process."
                break
        
    else:
        reason_to_stop = "Maximum attempts reached."
    
    # Compile the final result
    result = {
        "ReasonToStop": reason_to_stop,
        "FinalScore": final_score,
        "ChainOfThought": chain_of_thought,
        "CritiqueHistory": critique_history,
        "UserFeedbackIncorporated": user_feedback_incorporated if user_feedback_incorporated else None
    }
    
    return result


# Load API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")


# Replace 'your-openai-api-key' with your actual API key
generator_llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)

# Define JSON schema for critic LLM
json_schema = {
    "title": "CriticFeedback",
    "description": "Comprehensive feedback provided by the critic model to improve the reasoning chain effectively.",
    "type": "object",
    "properties": {
        "Critique": {
            "type": "string",
            "description": "A detailed critique of the reasoning chain, focusing on logical flow, gaps in reasoning, and overall coherence. Include specific examples of flaws in reasoning and suggestions for improvement."
        },
        "ClarifyingQuestions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific questions aimed at clarifying ambiguities or obtaining additional information that could help improve the content further."
        },
        "ReasoningScore": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "A score between 0 and 100 to rate the quality of the reasoning chain, where higher scores indicate better logical consistency and coherence."
        }
    },
    "required": ["Critique", "ClarifyingQuestions", "ReasoningScore"]
}

critic_llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key ).with_structured_output(json_schema)

input_prompt = input("Add here your request: ")

result = gan_feedback_loop(
    model_generator_llm=generator_llm,
    model_critic_llm=critic_llm,
    prompt=input_prompt,
    require_user_feedback=False,
    require_user_confirmation=False,
    min_score=85,
    max_attempts=4,
    stagnation_threshold=3
)


print("Reason to Stop:", result['ReasonToStop'])
print("Final Score:", result['FinalScore'])
print("Generated Chain of Thought:\n", result['ChainOfThought'])
