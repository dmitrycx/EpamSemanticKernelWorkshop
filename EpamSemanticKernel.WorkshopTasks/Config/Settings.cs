namespace EpamSemanticKernel.WorkshopTasks.Config;
using System.Text.Json;

public static class Settings
{
    private const string DefaultConfigFile = "Config/settings.json";
    private const string DefaultModel = "gpt-35-turbo";
    private const string ModelKey = "model";
    private const string SecretKey = "apikey";
    private const string ServiceIdKey = "serviceid";
    private const string EndpointKey = "endpoint";

    public static (string model, string azureEndpoint, string apiKey, string serviceId) LoadFromFile(
        string configFile = DefaultConfigFile,
        string model = DefaultModel)
    {
        try
        {
            if (string.IsNullOrEmpty(configFile) || !File.Exists(configFile))
            {
                Console.WriteLine("Invalid or missing configuration file path.");
                throw new ArgumentException("Invalid or missing configuration file path", nameof(configFile));
            }

            string json = File.ReadAllText(configFile);
            var settingsList = JsonSerializer.Deserialize<List<Dictionary<string, string>>>(json);

            if (settingsList == null || settingsList.Count == 0)
            {
                Console.WriteLine("No configurations found in the file.");
                throw new Exception("No configurations found");
            }

            var config = settingsList.FirstOrDefault(_ => _[ModelKey] == model);

            if (config == null)
            {
                Console.WriteLine($"Configuration for model '{model}' not found.");
                throw new Exception($"Configuration for model '{model}' not found");
            }

            // Validate and retrieve values
            string loadedModel = ValidateAndGet(config, ModelKey);
            string azureEndpoint = ValidateAndGet(config, EndpointKey);
            string apiKey = ValidateAndGet(config, SecretKey);
            string serviceId = ValidateAndGet(config, ServiceIdKey);

            if (serviceId == "none")
            {
                serviceId = "";
            }

            return (loadedModel, azureEndpoint, apiKey, serviceId);
        }
        catch (Exception e)
        {
            Console.WriteLine($"Something went wrong: {e.Message}");
            return ("", "", "", "");
        }
    }

    private static string ValidateAndGet(Dictionary<string, string> config, string key)
    {
        if (!config.TryGetValue(key, out var value) || string.IsNullOrEmpty(value))
        {
            Console.WriteLine($"Invalid or missing value for key '{key}'.");
            throw new ArgumentException($"Invalid or missing value for key '{key}'", nameof(key));
        }

        return value;
    }
}