using EpamSemanticKernel.WorkshopTasks.Config;
using Microsoft.SemanticKernel.Connectors.AI.OpenAI;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.AI.ChatCompletion;

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
    ServiceId = gpt4ServiceId
};

var prompt = @"{{$input}}
        One line TLDR with the fewest words.";
var summarize = kernel.CreateFunctionFromPrompt(prompt, aiRequestSettings);
        
var text = @"
        1st Law of Thermodynamics - Energy cannot be created or destroyed.
        2nd Law of Thermodynamics - For a spontaneous process, the entropy of the universe increases.
        3rd Law of Thermodynamics - A perfect crystal at zero Kelvin has zero entropy.";

Console.WriteLine(await kernel.InvokeAsync(summarize, new KernelArguments(text)));


// Chat
var input = "I want to find top-10 books about world history";
string skPrompt = @"ChatBot: How can I help you?
User: {{$input}}

---------------------------------------------

Return data requested by user: ";

var getShortIntentFunction  = kernel.CreateFunctionFromPrompt(skPrompt, aiRequestSettings);
var intentResult = await kernel.InvokeAsync(getShortIntentFunction, new KernelArguments(input));
Console.WriteLine(intentResult);

// Interactive chat
var chatHistory = new ChatHistory("You are a librarian, expert about books");

var message = chatHistory.Last();
Console.WriteLine($"{message.Role}: {message.Content}");

chatHistory.AddUserMessage("Hi, I'm looking for book suggestions");

message = chatHistory.Last();
Console.WriteLine($"{message.Role}: {message.Content}");

IChatCompletionService chatService = kernel.GetRequiredService<IChatCompletionService>(gpt35TurboServiceId);

var reply = await chatService.GetChatMessageContentAsync(chatHistory);
Console.WriteLine(reply);
chatHistory.AddSystemMessage(reply);

Func<string, Task> Chat = async (string input) => {
    // Save new message in the context variables
    chatHistory.AddUserMessage(input);

    var reply = await chatService.GetChatMessageContentAsync(chatHistory);

    chatHistory.AddSystemMessage(reply);
};

await Chat("I would like a non-fiction book suggestion about Greece history. Please only list one book.");
await Chat("that sounds interesting, what are some of the topics I will learn about?");
await Chat("Which topic from the ones you listed do you think the most popular?");

foreach (var message1 in chatHistory)
{
    Console.WriteLine($"{message1.Role}: {message1.Content}");
}