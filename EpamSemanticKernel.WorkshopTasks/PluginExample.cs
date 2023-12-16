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

        // Prompt
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
    }
}