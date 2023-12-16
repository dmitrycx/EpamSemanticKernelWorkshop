using EpamSemanticKernel.WorkshopTasks.Config;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.AI.OpenAI;
using Microsoft.SemanticKernel.Events;

namespace EpamSemanticKernel.WorkshopTasks;

internal class PipelineExample
{
    private PipelineExample()
    {
    }

    public static async Task RunAsync()
    {
         var builder = new KernelBuilder();
        // Loading models and setting up OpenAI chat completion services
        var (model1, azureEndpoint1, apiKey1, gpt35TurboServiceId) = Settings.LoadFromFile(model: "gpt-35-turbo");
        builder.AddAzureOpenAIChatCompletion(model1, model1, azureEndpoint1, apiKey1, serviceId: gpt35TurboServiceId);

        var (model2, azureEndpoint2, apiKey2, gpt4ServiceId) = Settings.LoadFromFile(model: "gpt-4-32k");
        builder.AddAzureOpenAIChatCompletion(model2, model2, azureEndpoint2, apiKey2, serviceId: gpt4ServiceId);

        var kernel = builder.Build();

        // Prompts initialization
        string findBooksPrompt = @"Please provide a list of the top {{$BooksNumber}} books that explore the fascinating realm of world history. 
It would be helpful if you could suggest books from the period {{$YearFrom}} to {{$YearTo}}. Show just Name and Author.";

        string translatePrompt = "Please translate the following text into {{$Lang}}. TEXT: {{$Input}}";

        // Create function for finding top books in world history
        KernelFunction findBooksFunction = kernel.CreateFunctionFromPrompt(
            findBooksPrompt,
            functionName: "TopBooks",
            description: "Retrieves a curated list of top books on world history within a specified time period.");

        // Create function for translating text
        KernelFunction translateFunction = kernel.CreateFunctionFromPrompt(
            translatePrompt,
            functionName: "Translate",
            description: "Translates the provided text into the specified language.");

        // Set up a pipeline of functions to be executed
        var pipeline = new KernelFunction[]
        {
            findBooksFunction,
            translateFunction
        };

        // Set execution settings for OpenAI service
        KernelArguments arguments = new KernelArguments
        {
            ExecutionSettings = new OpenAIPromptExecutionSettings
            {
                ServiceId = gpt35TurboServiceId,
                Temperature = 0
            }
        };

        // Set specific arguments for the findBooks function
        arguments.Add("BooksNumber", "3");
        arguments.Add("YearFrom", "1950");
        arguments.Add("YearTo", "1960");

        // Set specific arguments for the translate function
        arguments.Add("Lang", "Spanish");

        kernel.FunctionInvoking += Kernel_FunctionInvoking;
        kernel.FunctionInvoked += Kernel_FunctionInvoked;
        kernel.PromptRendering += Kernel_PromptRendering;
        kernel.PromptRendered += Kernel_PromptRendered;

        // Define event handlers
        void Kernel_PromptRendering(object? sender, PromptRenderingEventArgs e)
        {
            Console.WriteLine($"[PromptRendering]: \n\t // {e.Function.Description} \n\t {e.Function.Name} ({string.Join(":", e.Arguments)})");
        }

        void Kernel_PromptRendered(object? sender, PromptRenderedEventArgs e)
        {
            Console.WriteLine($"[PromptRendered]: \n\t{e.RenderedPrompt}");
        }

        void Kernel_FunctionInvoking(object? sender, FunctionInvokingEventArgs e)
        {
            Console.WriteLine($"[FunctionInvoking]: \n\t // {e.Function.Description} \n\t {e.Function.Name} ({string.Join(":", e.Arguments)})");
        }

        void Kernel_FunctionInvoked(object? sender, FunctionInvokedEventArgs e)
        {
            Console.WriteLine($"[FunctionInvoked]: \n\t {e.Function.Name}");
        }

        // Invoke each function in the pipeline and display the output
        var outputs = new List<string>();

        foreach (var item in pipeline)
        {
            var pipelineResult = await kernel.InvokeAsync(item, arguments);
            arguments["Input"] = pipelineResult;
            var valueAsString = pipelineResult.GetValue<string>();
            outputs.Add($"[{pipelineResult.Function.Name}]: {valueAsString}");
        }

        Console.WriteLine(string.Join("\n", outputs));
    }
}