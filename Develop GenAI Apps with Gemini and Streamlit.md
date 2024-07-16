When working with a generative text model, it can be difficult to coerce the LLM to give consistent responses in a structured format such as JSON. Function calling makes it easy to work with LLMs via prompts and unstructured inputs, and have the LLM return a structured response that can be used to call an external function.

You can think of function calling as a way to get structured output from user prompts and function definitions, use that structured output to make an API request to an external system, then return the function response to the LLM to generate a response to the user. In other words, function calling in Gemini extracts structured parameters from unstructured text or messages from users. In this example, you'll use function calling along with the chat modality in the Gemini model to help customers get information about products in the Google Store.
- Gemini can generate more than one function calls from a single prompt
- It can run these calls in parallel or serial and in a specific order based on whether it is serializable or parallel
### Examples:
1. Importing VertexAI libraries
```
import requests
from vertexai.generative_models import (
    Content,
    FunctionDeclaration,
    GenerationConfig,
    GenerativeModel,
    Part,
    Tool,
)
```
2. Defining functions:
```
get_product_info = FunctionDeclaration(
    name="get_product_info",
    description="Get the stock amount and identifier for a given product",
    parameters={
        "type": "object",
        "properties": {
            "product_name": {"type": "string", "description": "Product name"}
        },
    },
)

get_store_location = FunctionDeclaration(
    name="get_store_location",
    description="Get the location of the closest store",
    parameters={
        "type": "object",
	    "properties": {
	      "location": {
	        "type": "string", "description": "Location"
	      }
	    },
    },
)

place_order = FunctionDeclaration(
    name="place_order",
    description="Place an order",
    parameters={
        "type": "object",
        "properties": {
            "product": {"type": "string", "description": "Product name"},
            "address": {"type": "string", "description": "Shipping address"},
        },
    },
)
```
3. Define a tool that allows the Gemini model to select from the set of functions
```
retail_tool = Tool(
    function_declarations=[
        get_product_info,
        get_store_location,
        place_order,
    ],
)
```
4. Initialize the Gemini model with Function-calling
```
model = GenerativeModel(
    "gemini-1.5-pro-001",
    generation_config=GenerationConfig(temperature=0),
    tools=[retail_tool],
)
chat = model.start_chat()
```
5. Start the conversation
```
prompt = """
Do you have the Pixel 8 Pro in stock?
"""

response = chat.send_message(prompt)
response.candidates[0].content.parts[0]
```
6. This is what Gemini responds with
```
function_call {
  name: "get_product_info"
  args {
    fields {
      key: "product_name"
      value {
        string_value: "Pixel 8"
      }
    }
  }
}
```
7. Make an API request to generate a response
```
# Using a simulated response here
api_response = {"sku": "GA04834-US", "in_stock": "yes"}
```
8. Provide the response from API and generate a response for the user
```
response = chat.send_message(
    Part.from_function_response(
        name="get_product_info",
        response={
            "content": api_response,
        },
    ),
)
response.text
	```