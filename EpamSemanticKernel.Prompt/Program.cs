using Azure;
using Azure.AI.OpenAI;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.AI.OpenAI;

var builder = new KernelBuilder();
var client = new OpenAIClient(new Uri("https://ai-proxy.lab.epam.com"), new AzureKeyCredential(""));
builder.AddAzureOpenAIChatCompletion(
    "gpt-35-turbo",           // Azure OpenAI Deployment Name
    "https://ai-proxy.lab.epam.com", //Azure OpenAI Endpoint
    client);                                // Azure OpenAI Client
var kernel = builder.Build();

var prompt = @"{{$input}}
        One line TLDR with the fewest words.";
var summarize = kernel.CreateFunctionFromPrompt(prompt, executionSettings: new OpenAIPromptExecutionSettings { MaxTokens = 100 });
        
var text = @"
        1st Law of Thermodynamics - Energy cannot be created or destroyed.
        2nd Law of Thermodynamics - For a spontaneous process, the entropy of the universe increases.
        3rd Law of Thermodynamics - A perfect crystal at zero Kelvin has zero entropy.";

Console.WriteLine(await kernel.InvokeAsync(summarize, new KernelArguments(text)));





