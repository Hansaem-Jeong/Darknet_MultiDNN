#include <stdio.h>
#include <time.h>

/*
typedef enum KindofLayer {


};
*/
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

typedef struct MULTI_DNN {

    int on;
    int release;

    char* name;
    char* type;
    int numberof; // layer
//    enum KindofLayer kind; 

    int prior;
//    bool preemptive; // 1: enabled, 0: disabled
//    float period_msec; 
    struct timespec period;
//    float deadline;

    int lastlayer;
//    context;

    double release_time; 
    double complete_time;
} MultiDNN;

void *classification_thread(void *arg);
void *detection_thread(void *arg);

void *demo_classification_thread(void *arg);

void run_multidnn(int argc, char **argv);
