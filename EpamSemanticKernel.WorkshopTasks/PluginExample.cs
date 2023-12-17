using EpamSemanticKernel.WorkshopTasks.Config;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.AI.OpenAI;

namespace EpamSemanticKernel.WorkshopTasks;
using Microsoft.SemanticKernel.Plugins.Core;

public class PluginExample
{
    private PluginExample()
    {
    }
    public static async Task RunAsync()
    {
        var builder = new KernelBuilder();

        // default plugin
        var (model1, azureEndpoint1, apiKey1, gpt35TurboServiceId) = Settings.LoadFromFile(model: "gpt-35-turbo");
        builder.AddAzureOpenAIChatCompletion(model1, model1, azureEndpoint1, apiKey1, serviceId: gpt35TurboServiceId);

        var (model2, azureEndpoint2, apiKey2, gpt4ServiceId) = Settings.LoadFromFile(model: "gpt-4-32k");
        builder.AddAzureOpenAIChatCompletion(model2, model2, azureEndpoint2, apiKey2, serviceId: gpt4ServiceId);

        var (model3, _, apiKey3, hfServiceId) = Settings.LoadFromFile(model: "google/flan-t5-xxl");
        builder.AddHuggingFaceTextGeneration(model3, apiKey: apiKey3, serviceId:hfServiceId);

        var kernel = builder.Build();
        
        // Set execution settings for OpenAI service
        var arguments = new KernelArguments
        {
            ExecutionSettings = new OpenAIPromptExecutionSettings
            {
                ServiceId = hfServiceId,
                Temperature = 0
            }
        };
        
        var timeFunctions = kernel.ImportPluginFromObject(new TimePlugin(), "time");

        arguments.Add("input", "100");

        var result = await kernel.InvokeAsync(timeFunctions["daysAgo"], arguments);

        Console.WriteLine(result);
        
        // custom plugin
        var customPluginFunctions = kernel.ImportPluginFromObject(new CustomPlugin(kernel, serviceId: gpt4ServiceId));
        Console.WriteLine($"{string.Join("\n", customPluginFunctions.Select(_ => $"[{_.Name}] : {_.Description}"))}");

        Func<string, string, Task> TranslateAsync = async (string input, string lang) =>
        {
            // Save new message in the context variables
            arguments["input"] = input;
            arguments["lang"] = lang;

            var answer = await customPluginFunctions["Translate"].InvokeAsync(kernel, arguments);
        };

        arguments["history"] = string.Empty;

        Func<string, Task> ChatAsync = async (string input) =>
        {
            // Save new message in the context variables
            arguments["message"] = input;

            var answer = await customPluginFunctions["Chat"].InvokeAsync(kernel, arguments);
        };

        // Set specific arguments for the findBooks function
        arguments.Add("BooksNumber", "10");
        arguments.Add("YearFrom", "1900");
        arguments.Add("YearTo", "2000");
        
        var sourceInput = "Hi, I'm looking for the best historical book suggestions top-{{$BooksNumber}} from {{$YearFrom}} to {{$YearTo}}";

        await TranslateAsync(sourceInput, "Italian");
        await TranslateAsync(sourceInput, "Russian");
        await TranslateAsync(sourceInput, "Chinese");

        await ChatAsync(sourceInput);
        await ChatAsync("And from these books find top-10 the most interesting facts");
        await ChatAsync("Describe he main idea of selected facts");

        Console.WriteLine($"[Chat History]: {arguments["history"]}");
    }
}