using Microsoft.SemanticKernel;
using System.ComponentModel;

public class CustomPlugin
{
	private readonly Kernel _kernel;
	private readonly IPromptTemplateFactory _promptTemplateFactory;
	private readonly string _serviceId;

	public CustomPlugin(Kernel kernel, IPromptTemplateFactory promptTemplateFactory = null, string serviceId = "gpt35TurboServiceId")
	{
		_kernel = kernel;
		_promptTemplateFactory = promptTemplateFactory ?? new KernelPromptTemplateFactory();
		_serviceId = serviceId;
	}

	[KernelFunction("Chat"), Description("Initiates a chat with the ChatBot.")]
	public async Task<string> ChatAsync(KernelArguments arguments)
	{
		var prompt = @"
        Today is: {{time.Date}}
        Current time is: {{time.Time}}

        ChatBot can have a conversation with you about any topic.
            It can give detailed answer or say 'I don't know' if it does not have an answer.

            {{$history}}
            User: {{$message}}
            ChatBot:";

		var renderedPrompt = await _promptTemplateFactory.Create(new PromptTemplateConfig(prompt)).RenderAsync(_kernel, arguments);

		var skFunction = _kernel.CreateFunctionFromPrompt(
			promptTemplate: renderedPrompt,
			functionName: nameof(ChatAsync),
			description: "Complete the prompt.");

		var resultAsString = string.Empty;
		try
		{
			var result = await skFunction.InvokeAsync(_kernel, arguments);
			resultAsString = result.GetValue<string>();
		}
		catch (Exception exception)
		{
			Console.WriteLine(exception);
		}

		// Append the new interaction to the chat history
		string history = $"{arguments["history"]}";
		history += $"\nUser: {arguments["message"]}\nSuggestions: {resultAsString}\n";
		return history;
	}
	
	[KernelFunction("Translate"), Description("Initiates a translation task with the ChatBot by requesting the translation of a given text to a specified language.")]
	public async Task<string> TranslateAsync(KernelArguments arguments)
	{
		var prompt = @"You are the best linguist. Please translate the given TEXT to {{$lang}}. TEXT: {{$Input}}";

		var renderedPrompt = await _promptTemplateFactory.Create(new PromptTemplateConfig(prompt)).RenderAsync(_kernel, arguments);

		var skFunction = _kernel.CreateFunctionFromPrompt(
			promptTemplate: renderedPrompt,
			functionName: nameof(TranslateAsync),
			description: "Complete the prompt.");

		var result = await skFunction.InvokeAsync(_kernel, arguments);
		return result?.GetValue<string>() ?? string.Empty;
	}
}