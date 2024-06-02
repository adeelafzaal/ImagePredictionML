using ImagePredictionML.classes;
using Microsoft.Extensions.Options;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace ImagePredictionML.Classes
{
    public interface IImageClassifier
    {
        ITransformer GenerateModel(MLContext mlContext);
        PredictionOutput ClassifySingleImage(MLContext mlContext, ITransformer mode, string predictImage);
    }
    public class ImageClassifier : IImageClassifier
    {
        private readonly string _assetsPath;
        private readonly string _imagesFolder;
        private readonly string _trainTagsTsv;
        private readonly string _testTagsTsv;
        private readonly string _inceptionTensorFlowModel;

        private readonly AppSettings _appSettings;

        public ImageClassifier(IOptions<AppSettings> options) 
        { 
            _appSettings = options.Value;
            _assetsPath = _appSettings.TrainingDataPath;
            _imagesFolder = _assetsPath;
            _inceptionTensorFlowModel = _appSettings.InceptionModelPath;
            _trainTagsTsv = Path.Combine(_assetsPath, "tags.tsv");
            _testTagsTsv = Path.Combine(_assetsPath, "test-tags.tsv");
            _inceptionTensorFlowModel = Path.Combine(_inceptionTensorFlowModel, "tensorflow_inception_graph.pb");
    }

        public PredictionOutput ClassifySingleImage(MLContext mlContext, ITransformer model, string imagePath)
        {
            var imageData = new ImageInput()
            {
                ImagePath = imagePath
            };

            var predictor = mlContext.Model.CreatePredictionEngine<ImageInput, PredictionOutput>(model);
            var prediction = predictor.Predict(imageData);
            return prediction;
        }

        public ITransformer GenerateModel(MLContext mlContext)
        {
            IEstimator<ITransformer> pipeline = mlContext.Transforms
                .LoadImages(outputColumnName: "input", imageFolder: _imagesFolder, inputColumnName: nameof(ImageInput.ImagePath))
                .Append(mlContext.Transforms.ResizeImages(outputColumnName: "input", imageWidth: InceptionModelAttributes.ImageWidth, imageHeight: InceptionModelAttributes.ImageHeight, inputColumnName: "input"))
                .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input", interleavePixelColors: InceptionModelAttributes.ChannelsLast, offsetImage: InceptionModelAttributes.Mean))
                .Append(mlContext.Model.LoadTensorFlowModel(_inceptionTensorFlowModel)
                .ScoreTensorFlowModel(outputColumnNames: new[] { "softmax2_pre_activation" }, inputColumnNames: new[] { "input" }, addBatchDimensionInput: true))
                .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "LabelKey", inputColumnName: "Label"))
                .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: "LabelKey", featureColumnName: "softmax2_pre_activation"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedValue", "PredictedLabel"))
                .AppendCacheCheckpoint(mlContext);

            IDataView trainingData = mlContext.Data.LoadFromTextFile<ImageInput>(path: _trainTagsTsv, hasHeader: false);

            ITransformer model = pipeline.Fit(trainingData);

            IDataView testData = mlContext.Data.LoadFromTextFile<ImageInput>(path: _testTagsTsv, hasHeader: false);
            IDataView predictions = model.Transform(testData);

            IEnumerable<PredictionOutput> imagePredictionData = mlContext.Data.CreateEnumerable<PredictionOutput>(predictions, true);


            MulticlassClassificationMetrics metrics =
                mlContext.MulticlassClassification.Evaluate(predictions,
                  labelColumnName: "LabelKey",
                  predictedLabelColumnName: "PredictedLabel");


            return model;
        }

                
    }

    struct InceptionModelAttributes
    {
        public const int ImageHeight = 224;
        public const int ImageWidth = 224;
        public const float Mean = 117;
        public const float Scale = 1;
        public const bool ChannelsLast = true;
    }
}
