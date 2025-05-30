/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.sensingkit.flaaslib.ml;

import android.content.Context;
import android.util.Log;

import java.io.Closeable;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.channels.GatheringByteChannel;
import java.nio.channels.ScatteringByteChannel;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import org.tensorflow.lite.examples.transfer.api.AssetModelLoader;
import org.tensorflow.lite.examples.transfer.api.TransferLearningModel;
import org.tensorflow.lite.examples.transfer.api.TransferLearningModel.Prediction;

/**
 * App-layer wrapper for {@link TransferLearningModel}.
 *
 * <p>This wrapper allows to run training continuously, using start/stop API, in contrast to
 * run-once API of {@link TransferLearningModel}.
 */
public class TransferLearning implements Closeable {

    private TransferLearningModel model;

    private int epochs;
    private List<Float> epochResults;

    public TransferLearning(Context context, String modelDir, Collection<String> classes, int epochs) {

        model = new TransferLearningModel(
                new AssetModelLoader(context, modelDir), classes);

        this.epochs = epochs;
        this.epochResults = new ArrayList<>(epochs + 1);
    }


    // This method is thread-safe.
    public Future<Void> addSample(float[] input, String className) {
        return model.addSample(input, className);
    }

    // This method is thread-safe, but blocking.
    public Prediction[] predict(float[] input) {
        return model.predict(input);
    }

    public int getTrainBatchSize() {
        return model.getTrainBatchSize();
    }

    /**
     * Start training the model continuously until all epochs are done
     */
    public void startTraining() {
        try {
            Log.d("TransferLearning", "Starting training...");
            model.train(epochs, (epoch, loss) -> {
                if (Float.isNaN(loss) || Float.isInfinite(loss)) {
                    Log.e("TransferLearning", "Loss went bad at epoch " + epoch);
                }
                this.epochResults.add(loss);
            }).get();

        } catch (ExecutionException e) {
            throw new RuntimeException("Exception occurred during model training", e.getCause());
        } catch (InterruptedException e) {
            // no-op
        }
    }

    /**
     * Start SL training the model continuously for 1 epoch
     */
    public void startSLTraining() {
        try {
            Log.d("TransferLearning", "📦 Starting SL forward-pass preparation...");
            model.SLtrain().get();
            Log.d("TransferLearning", "✅ SL data ready.");
        } catch (ExecutionException e) {
            throw new RuntimeException("❌ Exception occurred during SL preparation", e.getCause());
        } catch (InterruptedException e) {
            // Thread was interrupted—handle gracefully if needed
            Thread.currentThread().interrupt();
            Log.e("TransferLearning", "⚠️ SL training interrupted");
        }
    }


    /** Frees all model resources and shuts down all background threads. */
    public void close() {
        model.close();
        model = null;
    }

    public void saveParameters(File file) {
        try {
            FileOutputStream out = new FileOutputStream(file);
            GatheringByteChannel gather = out.getChannel();
            model.saveParameters(gather);
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void saveSLParameters(File file) {
        try {
            FileOutputStream out = new FileOutputStream(file);
            GatheringByteChannel gather = out.getChannel();
            model.saveSLParameters(gather);  // Calls the method you already implemented
            out.close();
        } catch (IOException e) {
            Log.e("TransferLearningModel", "❌ Failed to save SL parameters", e);
            e.printStackTrace();
        }
    }


    public void loadParameters(File file) {
        try {
            FileInputStream inp = new FileInputStream(file);
            ScatteringByteChannel scatter = inp.getChannel();
            model.loadParameters(scatter);
            inp.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public List<Float> getEpochResults() {
        return this.epochResults;
    }

    public void applyDPNoise(float stddev) {
        if (model != null) {
            model.applyDPNoiseToWeights(stddev);
        }
    }


}