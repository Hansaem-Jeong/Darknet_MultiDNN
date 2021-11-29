#include <stdio.h>
#include <time.h>
#include "darknet.h"

/*
typedef enum KindofLayer {


};
*/

/* *** Choise Scheduling Type *** */
/* Non-Priority based Non-Preemptive Scheduling */
//#define BASIC_MULTIDNN
/* Priority based Non-Preemptive Scheduling */
#define PRIORITY_MULTIDNN
/* Priority based Preemptive Scheduling */
//#define PREEMPTION_MULTIDNN




//#define QUANTUM_MEASUREMENT
#define MEASUREMENT

#define QUANTUM_ITERATION 50000

#define MEASUREMENT_ITERATION 1050
#define MEAS_THRESHOLD 50

#define MEASUREMENT_PATH "measure"
#define MEASUREMENTD_FILE "/d_measure.csv"
#define MEASUREMENTC_FILE "/c_measure.csv"

#define DETECTOR_PERIOD 150
#define CLASSIFIER_PERIOD 150
// priority-based, classifier HIGH set
//#define PRIORITY_H


#define QUANTUM_SEC 0
#define QUANTUM_NSEC 500000 // 0.5ms
#define QUANTUM_PRIOR 5

#define ALEXNET_SEC 0
#define ALEXNET_NSEC 100000000 // 100ms
#define ALEXNET_PRIOR 5

#define DARKNET19_SEC 0
#define DARKNET19_NSEC 100000000 // 100ms
#define DARKNET19_PRIOR 6

#define DENSENET201_SEC 0
#define DENSENET201_NSEC 100000000 // 100ms
#define DENSENET201_PRIOR 7

#define CLASSIFIER_CNT 100


#define YOLOV2_SEC 0
#define YOLOV2_NSEC 150000000 // 150ms
#define YOLOV3_SEC 0
#define YOLOV3_NSEC 150000000 // 150ms
#define YOLOV4_SEC 0
#define YOLOV4_NSEC 150000000 // 150ms
#define YOLOV2T_SEC 0
#define YOLOV2T_NSEC 150000000 // 150ms
#define YOLOV3T_SEC 0
#define YOLOV3T_NSEC 150000000 // 150ms
#define YOLOV4T_SEC 0
#define YOLOV4T_NSEC 150000000 // 150ms

#define YOLO_PRIOR 10
#define DETECTOR_CNT 150

#define DET 0
#define CLA 1

#define THREAD_SLEEP_SEC 0
#define THREAD_SLEEP_NSEC 10000000  // 10ms

typedef struct DEMO_CLASSI {
    int idx;
    char *datacfg;
    char *cfgfile;
    char *weightfile;
    int cam_index;
    const char *filename;
    int benchmark;
    int benchmark_layers;
} DemoClassi;

typedef struct DEMO_DETECTOR {
    int idx;
    char *cfgfile;
    char *weightfile;
    float thresh;
    float hier_thresh;
    int cam_index;
    const char *filename;
    char **names;
    int classes;
    int avgframes;
    int frame_skip;
    char *prefix;
    char *out_filename;
    int mjpeg_port;
    int dontdraw_bbox;
    int json_port;
    int dont_show;
    int ext_output;
    int letter_box;
    int time_limit_sec;
    char *http_post_host;
    int benchmark;
    int benchmark_layers; 
} DemoDetector;

typedef struct ARGS {
    int idx;
    int argc;
    char **argv;
} Args;
/*
typedef struct IMAGE_FRAME {
    image frame;
    int sequence;
    double timestamp;

} ImageFrame;
*/ // in include/darknet.h


typedef struct MULTI_DNN {

    volatile int on;
    volatile int release;

    DNN_Info info;

    //char* name;
    //char* type;
    int numberof; // layer
//    enum KindofLayer kind; 

    //int prior;
//    bool preemptive; // 1: enabled, 0: disabled
//    float period_msec; 
    struct timespec period;
//    float deadline;

    int lastlayer;
//    context;


    ImageFrame detect_section;
    ImageFrame display_section; 
    float *prediction;

    int count;

    double frame_timestamp[MEASUREMENT_ITERATION];
    double release_period[MEASUREMENT_ITERATION];
    double release_time[MEASUREMENT_ITERATION]; 
    double onrunning_time[MEASUREMENT_ITERATION];
    double complete_time[MEASUREMENT_ITERATION];

    double before_prediction[MEASUREMENT_ITERATION];
    double after_prediction[MEASUREMENT_ITERATION];

    double display_start[MEASUREMENT_ITERATION];
    double display_end[MEASUREMENT_ITERATION];

} MultiDNN;

double multi_get_wall_time();
void *demo_detector_thread(void *arg);

void *demo_classification_thread(void *arg);

void *multi_fetch_in_thread(void *ptr);
void *multi_detect_in_thread(void *ptr);
void *multi_display_in_thread_sync(void *ptr);
void *multi_display_in_thread(void *arg);

void run_multidnn(int argc, char **argv);
