package org.sensingkit.flaaslib.workers;

import android.content.Context;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.work.Data;
import androidx.work.WorkerParameters;

import com.google.gson.JsonArray;

import org.sensingkit.flaaslib.FLaaSLib;
import org.sensingkit.flaaslib.dataset.cifar.CIFAR10BatchFileParser;
import org.sensingkit.flaaslib.enums.App;
import org.sensingkit.flaaslib.enums.DatasetType;
import org.sensingkit.flaaslib.enums.TrainingMode;
import org.sensingkit.flaaslib.ml.TransferLearning;
import org.sensingkit.flaaslib.utils.PerformanceCheckpoint;
import org.sensingkit.flaaslib.utils.PersistentStore;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutionException;



public class LocalTrainWorker extends AbstractFLaaSWorker {

    @SuppressWarnings("unused")
    private static final String TAG = LocalTrainWorker.class.getSimpleName();

    // Training variables
    TransferLearning tl;

    public LocalTrainWorker(@NonNull Context context, @NonNull WorkerParameters workerParams) {
        super(context, workerParams);
    }

    @NonNull
    @Override
    public Result doWork() {

        // performance measurement
        PerformanceCheckpoint totalPerformance = new PerformanceCheckpoint();

        // <-- Start
        totalPerformance.start();

        // prepare failure result
        Result failureResult = getFailureResult();

        // get context
        Context context = getApplicationContext();

        //Log.d(TAG, "Raw inputData: " + getInputData().toString());

        // get input data
        int backendRequestID = getInputData().getInt(KEY_BACKEND_REQUEST_ID_ARG, -1);
        int projectId = getInputData().getInt(KEY_PROJECT_ID_ARG, -1);
        int round = getInputData().getInt(KEY_ROUND_ARG, -1);
        String dataset = getInputData().getString(KEY_DATASET_ARG);
        DatasetType datasetType = DatasetType.fromName(getInputData().getString(KEY_DATASET_TYPE_ARG));
        int epochs = getInputData().getInt(KEY_EPOCHS_ARG, -1);
        int seed = getInputData().getInt(KEY_SEED_ARG, -1);
        String username = getInputData().getString(KEY_USERNAME_ARG);
        if (username == null) {
            return Result.failure();
        }
        int maxSamples = getInputData().getInt(KEY_MAX_SAMPLES_ARG, -1);
        String model = getInputData().getString(KEY_MODEL_ARG);
        if (model == null) {
            return Result.failure();
        }
        String trainingModeString = getInputData().getString(KEY_TRAINING_MODE_ARG);
        TrainingMode trainingMode = TrainingMode.fromValue(trainingModeString);
        long receivedTime = getInputData().getLong(AbstractFLaaSWorker.KEY_WORKER_SCHEDULED_TIME_ARG, -1);
        long validDate = getInputData().getLong(AbstractFLaaSWorker.KEY_REQUEST_VALID_DATE_ARG, -1);
        int localDP = getInputData().getInt(AbstractFLaaSWorker.KEY_DP_ARG, 0);
        Log.d(TAG, "Local DP flag: " + localDP);
        float epsilon = getInputData().getFloat(AbstractFLaaSWorker.KEY_EPSILON_ARG, 1.0f);
        float delta = getInputData().getFloat(AbstractFLaaSWorker.KEY_DELTA_ARG, 1e-5f);
        boolean useSplitLearning = getInputData().getBoolean(AbstractFLaaSWorker.KEY_USE_SPLIT_LEARNING, false);
        Log.d(TAG, "Use Split Learning: " + useSplitLearning);


        // init stats
        String statsJson = getInputData().getString(KEY_STATS_ARG);
        loadStats(statsJson);

        // add worker started after X secs
        float workerStarted = (float)((System.nanoTime() - receivedTime) / 1e9);
        addProperty("local_training_worker", "worker_started_after", workerStarted);

        float[] durationsStats = getInputData().getFloatArray(KEY_DURATIONS_ARG);
        if (durationsStats != null) {
            for (App app : App.rgbApps()) {
                addProperty("download_weights_worker", app.getName() + "_request_duration", durationsStats[app.getId()]);
            }
        }

        // init random with computed seed
        this.random = createRandom(seed, round, username);

        // init model, load weights and train
        Log.d(TAG, "Conducting training...");
        if (!conductTraining(context, trainingMode, projectId, round, dataset, datasetType, maxSamples, model, epochs, localDP, epsilon, delta, useSplitLearning)) {
            Log.e(TAG, "Conduct training failed.");
            return failureResult;
        }

        // End and report -->
        totalPerformance.end();
        addProperty("local_training_worker", "worker_duration", totalPerformance.getDuration());

        // add attempt counter
        addProperty("local_training_worker", "attempt", getRunAttemptCount());

        // Get stats file
        String jsonString = getStatsInJsonString();

        // if we are in an RGB app and requested to join models at FLaaS level
        if (trainingMode == TrainingMode.JOINT_MODELS) {

            // send weights
            int requestId = getInputData().getInt(KEY_REQUEST_ID_ARG, -1);
            FLaaSLib.sendWeights(context, App.FLAAS, requestId, projectId, round, jsonString);

            // delete weights
            String prefix = projectId + "_" + round + "_";
            File globalModelFile = new File(context.getFilesDir(), prefix + FLaaSLib.MODEL_WEIGHTS_FILENAME);
            if (globalModelFile.exists()) //noinspection ResultOfMethodCallIgnored
                globalModelFile.delete();

            return Result.success();
        }
        // else, we are in Baseline mode, so continue

        // Build output
        Data output = new Data.Builder()
                .putInt(KEY_BACKEND_REQUEST_ID_ARG, backendRequestID)
                .putInt(KEY_PROJECT_ID_ARG, projectId)
                .putInt(KEY_ROUND_ARG, round)
                .putLong(KEY_WORKER_SCHEDULED_TIME_ARG, System.nanoTime())
                .putString(KEY_STATS_ARG, jsonString)
                .putLong(KEY_REQUEST_VALID_DATE_ARG, validDate)
                .build();

        return Result.success(output);
    }

    @SuppressWarnings("unused")
    private boolean conductTraining(Context context, TrainingMode trainingMode, int projectId, int round, String dataset, DatasetType datasetType, int maxSamples, String model, int epochs, int localDP, float epsilon, float delta, boolean useSplitLearning) {

        // performance measurement
        PerformanceCheckpoint loadWeightsPerformance = new PerformanceCheckpoint();
        PerformanceCheckpoint loadSamplesPerformance = new PerformanceCheckpoint();
        PerformanceCheckpoint trainPerformance = new PerformanceCheckpoint();

        // init model
        this.tl = new TransferLearning(context, model, Arrays.asList(CIFAR10BatchFileParser.getClasses()), epochs);

        // <-- Start
        loadWeightsPerformance.start();

        // load weights
        String prefix = projectId + "_" + round + "_";
        File globalModelFile = new File(context.getFilesDir(), prefix + FLaaSLib.MODEL_WEIGHTS_FILENAME);

        final long MAX_EXPECTED_HEAD_SIZE = 3 * 1024 * 1024; // 3MB

        if (!globalModelFile.exists()) {
            Log.w(TAG, "❌ Global model file does not exist, skipping loadParameters()");
        } else {
            long fileSizeBytes = globalModelFile.length();
            Log.d(TAG, "📄 Global model file size: " + fileSizeBytes + " bytes");

            if (fileSizeBytes > MAX_EXPECTED_HEAD_SIZE) {
                Log.w(TAG, "⚠️ Skipping loadParameters(): file too large. Assuming full model or invalid.");
            } else {
                Log.d(TAG, "📥 Loading head weights from file...");
                this.tl.loadParameters(globalModelFile);
            }
        }

        // End and report -->
        loadWeightsPerformance.end();
        addProperty("local_training_worker", "load_weights_duration", loadWeightsPerformance.getDuration());

        File[] sampleFiles;
        if (trainingMode == TrainingMode.BASELINE) {
            // Use the tmp files we stored when login. Its a baseline after all
            sampleFiles = new File[3];
            sampleFiles[0] = new File(context.getFilesDir(), App.RED.getName() + "_" + datasetType.getFilename());
            sampleFiles[1] = new File(context.getFilesDir(), App.GREEN.getName() + "_" + datasetType.getFilename());
            sampleFiles[2] = new File(context.getFilesDir(), App.BLUE.getName() + "_" + datasetType.getFilename());
        }
        else if (trainingMode == TrainingMode.JOINT_SAMPLES) {
            // Use the PersistentStore
            sampleFiles = new File[3];
            sampleFiles[0] = PersistentStore.getSamplesFile(context, App.RED, projectId, round, datasetType);
            sampleFiles[1] = PersistentStore.getSamplesFile(context, App.GREEN, projectId, round, datasetType);
            sampleFiles[2] = PersistentStore.getSamplesFile(context, App.BLUE, projectId, round, datasetType);
        }
        else {  // JOINT MODELS

            // This is local in an RGB app, so, the usual file
            sampleFiles = new File[1];
            sampleFiles[0] = new File(context.getFilesDir(), datasetType.getFilename());
        }

        // <-- Start
        loadSamplesPerformance.start();

        // load samples from filenames
        for (File samplesFile : sampleFiles) {
            // CIFAR10BatchFileParser dataManager;
            CIFAR10BatchFileParser dataManager = null;
            try {
                dataManager = new CIFAR10BatchFileParser(samplesFile, 0, 224);

               // add samples
                for (int i = 0; i < maxSamples; i++) {
                    if (!dataManager.hasNext()) {
                        Log.e(TAG, "Not enough samples.");
                        break;
                    }
                    // get next
                    dataManager.next();

                    // get data
                    float[] data = dataManager.getData(random.nextBoolean());
                    int label = dataManager.getLabel();

                    try {
                        tl.addSample(data, CIFAR10BatchFileParser.getClass(label)).get();
                    } catch (ExecutionException | InterruptedException e) {
                        e.printStackTrace();
                        return false;
                    }
                }

            } catch (IOException e) {
                e.printStackTrace();
                return false;

            } finally {
                if (dataManager != null) {
                    dataManager.close(); // ensure closure even on failure
                }
            }
            // close dataManager (not needed any more)
            // dataManager.close();
        }

        // End and report -->
        loadSamplesPerformance.end();
        addProperty("local_training_worker", "load_samples_duration", loadSamplesPerformance.getDuration());

        // <-- Start
        trainPerformance.start();

        // Train
        if (useSplitLearning) {
            tl.startSLTraining();
            // End and report -->
            trainPerformance.end();
            addProperty("local_training_worker", "training_duration", trainPerformance.getDuration());

            // delete the received samples
            PersistentStore.clearAllSamples(context, projectId, round, datasetType);

            // save weights (replace existing file)
            if (globalModelFile.exists()) //noinspection ResultOfMethodCallIgnored
                globalModelFile.delete();

            tl.saveSLParameters(globalModelFile);

            // close TL
            tl.close();
            tl = null;


        } else {
            tl.startTraining();
            // End and report -->
            trainPerformance.end();
            addProperty("local_training_worker", "training_duration", trainPerformance.getDuration());

            // delete the received samples
            PersistentStore.clearAllSamples(context, projectId, round, datasetType);

            // Get training results
            List<Float> epochResults = tl.getEpochResults();

            // Check for NaN losses
            for (int i = 0; i < epochResults.size(); i++) {
                Float loss = epochResults.get(i);
                if (loss == null || Float.isNaN(loss)) {
                    Log.w(TAG, "Epoch " + i + " produced NaN or null. Training failed.");
                    return false;
                }
            }

            // Add training epochs in performance results
            JsonArray epochsArray = new JsonArray(epochResults.size());
            // JsonArray epochsArray = new JsonArray(epochs);
            for (Float epoch : epochResults) {
                epochsArray.add(epoch);
            }
            addJsonArray("local_training_worker", "epochs", epochsArray);

            // save weights (replace existing file)
            if (globalModelFile.exists()) //noinspection ResultOfMethodCallIgnored
                globalModelFile.delete();

            if (localDP == 1) {
                Log.d(TAG, "Applying DP to TransferLearning weights...");
                if (epsilon <= 0 || delta <= 0 || delta >= 1) {
                    Log.e(TAG, "❌ Invalid epsilon or delta values for DP. Skipping DP noise.");
                    return false;  // Or optionally continue without DP: just skip applyDPNoise
                }
                // set sensitivity = 1, because the range of model parameters is in [-1,1]
                float stddev = computeGaussianStdDev(1, epsilon, delta);
                Log.d(TAG, "✅ Computed DP Gaussian noise stddev: " + stddev);
                tl.applyDPNoise(stddev);
            }

            tl.saveParameters(globalModelFile);

            // close TL
            tl.close();
            tl = null;
        }
        return true;
    }

    public float computeGaussianStdDev(float sensitivity, float epsilon, float delta) {
        double numerator = sensitivity * Math.sqrt(2 * Math.log(1.25 / delta));
        return (float) (numerator / epsilon);
    }
}
