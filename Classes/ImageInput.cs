using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ImagePredictionML.classes
{
    public class ImageInput
    {
        [LoadColumn(0)]
        public string? ImagePath;

        [LoadColumn(1)]
        public string? Label;
    }


}
