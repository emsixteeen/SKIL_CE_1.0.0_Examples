package ai.skymind.skil.examples.modelserver.inference;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.net.URL;
import java.util.Arrays;

public class NormalizeUciData {

    private enum Normalizer {
        Standardize (new NormalizerStandardize()),
        MinMax (new NormalizerMinMaxScaler());

        private DataNormalization normalizer;

        private Normalizer(DataNormalization normalizer) {
            this.normalizer = normalizer;
        }

        public DataNormalization getNormalizer() {
            return normalizer;
        }
    }

    @Parameter(names = "--url", description = "URL to download from", required = false)
    private String downloadUrl = "https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/synthetic_control.data";

    @Parameter(names = "--normalizer", description = "Normalizer to use", required = false)
    private Normalizer dataNormalizer = Normalizer.Standardize;

    @Parameter(names = "--output", description = "Where to write data to", required = false)
    private String outputPath = "/tmp/synthetic_control.data-normalized.csv";

    public void run() throws Exception {
        File outputFile = new File(outputPath);
        if (outputFile.exists()) {
            throw new IllegalStateException(
                    String.format("cowardly refusing to overwrite output file %s", outputPath));
        }

        System.out.format("downloading from %s\n", downloadUrl);
        System.out.format("writing output to %s\n", outputPath);

        URL url = new URL(downloadUrl);
        String data = IOUtils.toString(url);
        String[] lines = data.split("\n");
        INDArray array = Nd4j.zeros(lines.length, 60);

        for (int i=0; i<lines.length; i++ ) {
            String line = lines[i];

            String[] cols = line.split("\\s+");
            for (int j=0; j<cols.length; j++) {
                Double d = Double.parseDouble(cols[j]);
                array.putScalar(i, j, d);
            }
        }

        DataSet ds = new DataSet(array, array);
        DataSetIterator it = new ListDataSetIterator(ds.asList());
        DataNormalization normalizer = dataNormalizer.getNormalizer();

        normalizer.fit(it);
        it.setPreProcessor(normalizer);

        StringBuffer sb = new StringBuffer();

        while (it.hasNext()) {
            INDArray features = it.next(1).getFeatures();
            int[] shape = features.shape();

            if (!(shape[0] == 1 && shape[1] == 60)) {
                throw new IllegalStateException(String.format("wrong shape: %s", Arrays.toString(shape)));
            }

            for (int i=0; i<60; i++) {
                sb.append(features.getColumn(i));

                if (i < 59) {
                    sb.append(", ");
                }
            }

            sb.append("\n");
        }

        FileUtils.write(outputFile, sb);
        System.out.format("wrote normalized file to %s\n", outputPath);
    }

    public static void main(String... args) throws Exception {
        NormalizeUciData m = new NormalizeUciData();
        JCommander.newBuilder()
                .addObject(m)
                .build()
                .parse(args);

        m.run();
    }
}
