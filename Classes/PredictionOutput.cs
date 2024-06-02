using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ImagePredictionML.classes
{
    public class PredictionOutput : ImageInput
    {
        public float[]? Score;

        public string? PredictedValue;
    }
}
