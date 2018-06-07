package gfl.com.surfimpl;

import android.app.Activity;
import android.app.AlertDialog;
import android.app.ProgressDialog;
import android.content.Context;
import android.content.res.AssetManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Environment;
import android.os.Handler;
import android.os.Message;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.TextView;
import android.widget.ToggleButton;

import com.opencsv.CSVWriter;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.DMatch;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Scalar;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FastFeatureDetector;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.TooManyListenersException;
import java.util.stream.Collectors;

/**
 * Created by GFL
 */
public class MainActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "OCVSample::Activity";
    private int w, h;
    private CameraBridgeViewBase mOpenCvCameraView;
    private ToggleButton mToogleButton;
    TextView tvName;
    Scalar RED = new Scalar(255, 0, 0);
    Scalar GREEN = new Scalar(0, 255, 0);
    FeatureDetector detector;
    DescriptorExtractor descriptor;
    DescriptorMatcher matcher;
    Mat descriptors2, descriptors1;
    Mat img1;
    MatOfKeyPoint keypoints1, keypoints2;

    Long timestamp1;

    static Integer MAX_DATA_RECORDING = 10;
    Integer dataCounter = 0;
    List<String[]> data = new ArrayList<String[]>();

    int index = 0;

    static {
        if (!OpenCVLoader.initDebug())
            Log.d("ERROR", "Unable to load OpenCV");
        else
            Log.d("SUCCESS", "OpenCV loaded");
    }

    /**
     */
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                    try {
                        initializeOpenCVDependencies();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    /**
     */
    private void initializeOpenCVDependencies() throws IOException {
        mOpenCvCameraView.enableView();

        detector = FeatureDetector.create(FeatureDetector.AKAZE);
        descriptor = DescriptorExtractor.create(DescriptorExtractor.AKAZE);

        matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);

        img1 = new Mat();
        AssetManager assetManager = getAssets();
        InputStream istr = assetManager.open("a.jpeg");
        Bitmap bitmap = BitmapFactory.decodeStream(istr);
        Utils.bitmapToMat(bitmap, img1);
        Imgproc.cvtColor(img1, img1, Imgproc.COLOR_RGB2GRAY);
        img1.convertTo(img1, 0); //converting the image to match with the type of the cameras image

        descriptors1 = new Mat();
        keypoints1 = new MatOfKeyPoint();
        detector.detect(img1, keypoints1);
        descriptor.compute(img1, keypoints1, descriptors1);
    }

    /**
     */
    public MainActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /**
     * Called when the activity is first created.
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {

        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.tutorial1_activity_java_surface_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mToogleButton = (ToggleButton)findViewById(R.id.buttonEnable);
        mOpenCvCameraView.setCvCameraViewListener(this);
        tvName = (TextView) findViewById(R.id.text1);
        data.add(new String[]{"keypoints1", "keypoints2", "good_matches", "timestamp"});
    }

    /**
     */
    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    /**
     */
    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    /**
     * on destroy view
     */
    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    /**
     * @param width -  the width of the frames that will be delivered
     * @param height - the height of the frames that will be delivered
     */
    public void onCameraViewStarted(int width, int height) {
        w = width;
        h = height;
    }

    /**
     */
    public void onCameraViewStopped() {
        //export csv
        try {
            exportInCSV(this.data);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * @param aInputFrame camera stream
     * @return matrix of output img
     */
    public Mat recognize(Mat aInputFrame) {
        Imgproc.cvtColor(aInputFrame, aInputFrame, Imgproc.COLOR_RGB2GRAY);
        descriptors2 = new Mat();
        keypoints2 = new MatOfKeyPoint();
        detector.detect(aInputFrame, keypoints2);

        descriptor.compute(aInputFrame, keypoints2, descriptors2);

        // Matching
        MatOfDMatch matches = new MatOfDMatch();
        if (img1.type() == aInputFrame.type()) {
            matcher.match(descriptors1, descriptors2, matches);
        } else {
            return aInputFrame;
        }

        List<DMatch> matchesList = matches.toList();

        Double max_dist = 0.0;
        Double min_dist = 100.0;

        for (int i = 0; i < matchesList.size(); i++) {
            Double dist = (double) matchesList.get(i).distance;
            if (dist < min_dist)
                min_dist = dist;
            if (dist > max_dist)
                max_dist = dist;
        }

        LinkedList<DMatch> good_matches = new LinkedList<DMatch>();

        for (int i = 0; i < matchesList.size(); i++) {
            if (matchesList.get(i).distance <= (1.5 * min_dist))
                good_matches.addLast(matchesList.get(i));
        }

        MatOfDMatch goodMatches = new MatOfDMatch();
        goodMatches.fromList(good_matches);
        Mat outputImg = new Mat();
        MatOfByte drawnMatches = new MatOfByte();

        if (aInputFrame.empty() || aInputFrame.cols() < 1 || aInputFrame.rows() < 1) {
            return aInputFrame;
        }

        Features2d.drawMatches(img1, keypoints1, aInputFrame, keypoints2, goodMatches, outputImg, GREEN, RED, drawnMatches, Features2d.NOT_DRAW_SINGLE_POINTS);
        Imgproc.resize(outputImg, outputImg, aInputFrame.size());

        Long res = System.currentTimeMillis() - timestamp1;

        Log.i("time stamp inner: ", res.toString());

        if(dataCounter < MAX_DATA_RECORDING){
            String[] dataLine = new String[]{String.valueOf(keypoints1.toList().size()), String.valueOf(keypoints2.toList().size()), String.valueOf(good_matches.size()), String.valueOf(res)};
            data.add(dataLine);
            dataCounter++;
        }

        return outputImg;
    }

    /**
     *
     * @param inputFrame camera stream
     * @return recognize input frame with rgba method
     */
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        this.timestamp1 = System.currentTimeMillis();

        if(this.mToogleButton.isChecked())
            return recognize(inputFrame.rgba());
        else
            return inputFrame.rgba();
    }

    /**
     * @param data list of data to write in csv
     * @throws IOException if writer cannot be done
     */
    public void exportInCSV(List<String[]> data) throws IOException {
        File folder = new File(Environment.getExternalStorageDirectory()
                + "/METRICS");

        Log.i("debug", folder.getAbsolutePath());
        Log.i("debug", folder.getCanonicalPath());
        if (!folder.exists())
            folder.mkdir();

        final String filename = folder.toString() + "/" + "metrics.csv";

        try (FileOutputStream fos = new FileOutputStream(filename);
             OutputStreamWriter osw = new OutputStreamWriter(fos,
                     StandardCharsets.UTF_8);
             CSVWriter writer = new CSVWriter(osw)) {

            writer.writeAll(data);
        }
    }
}
