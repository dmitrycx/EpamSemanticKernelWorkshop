using EpamSemanticKernel.WorkshopTasks.Config;
using Microsoft.SemanticKernel.Connectors.AI.OpenAI;
using Microsoft.SemanticKernel;

var builder = new KernelBuilder();

// Prompt
var (model1, azureEndpoint1, apiKey1, gpt35TurboServiceId) = Settings.LoadFromFile(model: "gpt-35-turbo");
builder.AddAzureOpenAIChatCompletion(model1, model1, azureEndpoint1, apiKey1, serviceId: gpt35TurboServiceId);

var (model2, azureEndpoint2, apiKey2, gpt4ServiceId) = Settings.LoadFromFile(model: "gpt-4-32k");
builder.AddAzureOpenAIChatCompletion(model2, model2, azureEndpoint2, apiKey2, serviceId: gpt4ServiceId);

var kernel = builder.Build();

var aiRequestSettings = new OpenAIPromptExecutionSettings 
{
    ExtensionData = new Dictionary<string, object> { { "api-version", "2023-03-15-preview" } },
    ServiceId = gpt35TurboServiceId
};

var prompt = @"{{$input}}
        One line TLDR with the fewest words.";
var summarize = kernel.CreateFunctionFromPrompt(prompt, aiRequestSettings);
        
var text = @"
        1st Law of Thermodynamics - Energy cannot be created or destroyed.
        2nd Law of Thermodynamics - For a spontaneous process, the entropy of the universe increases.
        3rd Law of Thermodynamics - A perfect crystal at zero Kelvin has zero entropy.";

Console.WriteLine(await kernel.InvokeAsync(summarize, new KernelArguments(text)));