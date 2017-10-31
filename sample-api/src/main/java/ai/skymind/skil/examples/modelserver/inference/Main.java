package ai.skymind.skil.examples.modelserver.inference;

import java.util.Arrays;

public class Main {
    public static void main (String... args) throws Exception {
        final String usage = "usage: inference | directInference | normalizeUciData";

        if (args.length > 0) {
            String sub = args[0];
            String[] subArgs = Arrays.copyOfRange(args, 1, args.length);

            if (sub.equalsIgnoreCase("inference")) {
                ModelServerInferenceExample.main(subArgs);
            } else if (sub.equalsIgnoreCase("directInference")) {
                ModelServerDirectInferenceExample.main(subArgs);
            } else if (sub.equalsIgnoreCase("normalizeUciData")) {
                NormalizeUciData.main(subArgs);
            } else {
                System.out.println(String.format("unknown: %s", args[0]));
                System.out.println(usage);
            }
        } else {
            System.out.println(usage);
        }
    }
}
