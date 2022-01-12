#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "multidnn.h"
#ifdef WIN32
#include <time.h>
#include "gettimeofday.h"
#else
#include <sys/time.h>
#endif

#include "network.h"
#include "utils.h"
#include "parser.h"
#include "option_list.h"
#include "blas.h"
#include "assert.h"
#include "dark_cuda.h"
#include "detection_layer.h"

#include "region_layer.h"
#include "cost_layer.h"
#include "box.h"
#include "image.h"
#include "darknet.h"
#include "http_stream.h"

#include "nvToolsExt.h"
nvtxRangeId_t nvtx_name_CLA;
nvtxRangeId_t nvtx_name_DET;

struct timespec thread_sleep = {THREAD_SLEEP_SEC, THREAD_SLEEP_NSEC};
//thread_sleep.tv_sec = THREAD_SLEEP_SEC;
//thread_sleep.tv_nsec = THREAD_SLEEP_NSEC;

int tmp_c = 0;


#ifdef QUANTUM_MEASUREMENT

double meas_quantum_time_array[QUANTUM_ITERATION];

#endif

volatile system_exit_flag;

volatile int webcam_is_ok;

volatile int det_count = 0;
volatile int cla_count = 0;

ImageFrame once_frame[3];
ImageFrame fetch_frame[10];
ImageFrame classi_frame[10];
int fetch_idx;
int detect_idx;
int classification_idx;


pthread_mutex_t main_lock = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t tmp_lock = PTHREAD_MUTEX_INITIALIZER;

volatile MultiDNN dnn_buffer[10];
pthread_t demo_thread[10];

/*****  DemoDetector *****/
static char **demo_names;
static image **demo_alphabet;
static int demo_classes;

static int nboxes = 0;
static detection *dets = NULL;

static network net[3];
static network net_c[10];
static image in_s ;
static image det_s;

static cap_cv *cap;
static float fps = 0;
static float demo_thresh = 0;
static int demo_ext_output = 0;
static long long int frame_id = 0;
static int demo_json_port = -1;
static bool demo_skip_frame = false;


static int avg_frames;
static int demo_index = 0;
static mat_cv** cv_images;

mat_cv* in_img;
mat_cv* det_img;
mat_cv* show_img;

static volatile int flag_exit;
static int letter_box = 0;

static const int thread_wait_ms = 1;
static volatile int run_fetch_in_thread = 0;
static volatile int run_detect_in_thread[3] = { 0 };
static volatile int run_classification_in_thread[10] = { 0 };

static volatile int run_prediction_gpu = 0;
/*************************/

double multi_get_wall_time()
{
    
    struct timespec after_boot;
    clock_gettime(CLOCK_MONOTONIC, &after_boot);
    return (after_boot.tv_sec*1000 + after_boot.tv_nsec*0.000001);
    
}


void *demo_detector_thread(void *arg);

void *demo_classification_thread(void *arg)
{
#ifdef OPENCV
    printf("- Classifier Demo\n");

    DemoClassi *argm = (DemoClassi *)arg;
    int idx = argm->idx;
    char *datacfg = argm->datacfg;
    char *cfgfile = argm->cfgfile;
    char *weightfile = argm->weightfile;
    int cam_index = argm->cam_index;
    const char *filename = argm->filename;
    int benchmark = argm->benchmark;
    int benchmark_layers = argm->benchmark_layers;
    
    printf("        idx: %d\n", idx);
    printf("    Datacfg: %s\n", datacfg);
    printf("    Cfgfile: %s\n", cfgfile);
    printf(" Weightfile: %s\n", weightfile);
    printf("  Cam_index: %d\n", cam_index);
    printf("   Filename: %s\n", filename);

//    network net_c = parse_network_cfg_custom(cfgfile, 1, 0);
    net_c[idx] = parse_network_cfg_custom(cfgfile, 1, 0);
    if(weightfile){
        load_weights(&net_c[idx], weightfile);
    }

    printf("---- Success Classifier Network setting\n");
    printf("#### Thesis Classifier -> Network net.n: %d\n", net_c[idx].n); 

    net_c[idx].benchmark_layers = benchmark_layers;
    set_batch_network(&net_c[idx], 1);
    list *options = read_data_cfg(datacfg);

    fuse_conv_batchnorm(net_c[idx]);
    calculate_binary_weights(net_c[idx]);

    dnn_buffer[idx].on = 1;

    srand(2222222);
    
#endif
}

void *multi_classification_in_thread(void *ptr)
{
    printf("thesis check start classification thread\n");
    int cla_thread_idx = *((int *)ptr);
    printf("thesis check 2 classification thread, idx : %d\n", cla_thread_idx);
    while (!custom_atomic_load_int(&flag_exit)) {
        while (!custom_atomic_load_int(&run_classification_in_thread[cla_thread_idx])) {
            if (custom_atomic_load_int(&flag_exit)) return 0;
            this_thread_yield();
        }
        int this_idx = cla_thread_idx;
        printf("Thesis Check Start Classification Thread\n");
        //dnn_buffer[this_idx].detect_section = classi_frame[classification_idx];
        dnn_buffer[this_idx].detect_section = once_frame[0];
        dnn_buffer[this_idx].frame_timestamp[cla_count] = dnn_buffer[this_idx].detect_section.timestamp;
/*        
        det_s = in_s;
        det_img = in_img;
*/


        image in_c, in;
        in_c = dnn_buffer[this_idx].detect_section.classi_frame;
        //release_mat(dnn_buffer[this_idx].detect_section.classi_frame);
/*
        dnn_buffer[CLA].detect_section.classi_frame = in_c;
*/
        int tmp_count = cla_count;

//        dnn_buffer[this_idx].before_prediction[tmp_count] = multi_get_wall_time();
        double start_pred = multi_get_wall_time();

        //float *predictions = network_predict(net_c, in_c.data);
//        nvtx_name_CLA = nvtxRangeStartA("MultiDNN - Classifier");
        //float *predictions = multi_network_predict(net_c[cla_thread_idx], in_c.data, dnn_buffer[this_idx].info);
        float *predictions = multi_network_predict(net_c[cla_thread_idx], in_c.data, &dnn_buffer[this_idx]);

        dnn_buffer[this_idx].overhead[dnn_buffer[this_idx].count] = dnn_buffer[this_idx].overhead_time[dnn_buffer[this_idx].count]-dnn_buffer[this_idx].release_time[dnn_buffer[this_idx].count];
        printf(" ==-=-=-== THESIS %lf\n", dnn_buffer[this_idx].overhead);        

//        nvtxRangeEnd(nvtx_name_CLA);

//        dnn_buffer[this_idx].after_prediction[tmp_count] = multi_get_wall_time();
        dnn_buffer[this_idx].prediction_time[tmp_count] = multi_get_wall_time() - start_pred;
        printf("Classifier this_idx: %d, tmp_count: %d, prediction_time: %0.1lf\n",
        this_idx, tmp_count, dnn_buffer[this_idx].prediction_time[tmp_count]);

//        dnn_buffer[this_idx].prediction = predictions;


        dnn_buffer[this_idx].release = 1; // += 1?


        custom_atomic_store_int(&run_classification_in_thread[cla_thread_idx], 0);
    }

    printf("Thesis Out of Boundary Detect Thread\n");
    return 0;
}

void *multi_fetch_in_thread(void *ptr)
{
    printf("thesis check start fetch thread\n");
    printf("Thesis Check dnn_buffer[0].on : %d\n", dnn_buffer[0].on);
    int since_now = 1;
    while (!custom_atomic_load_int(&flag_exit)) {
        while(!dnn_buffer[1].on) {};
//        printf("thesis check in fetch thread while\n");
        
        int dont_close_stream = 0;    // set 1 if your IP-camera periodically turns off and turns on video-stream

// lock(), in_s : fetch image data
   
/* 
        if (letter_box)
            in_s = get_image_from_stream_letterbox(cap, net.w, net.h, net.c, &in_img, dont_close_stream);
        else
            in_s = get_image_from_stream_resize(cap, net.w, net.h, net.c, &in_img, dont_close_stream);
*/
/*
        classi_frame[fetch_idx] = fetch_frame[fetch_idx];
        detect_idx = fetch_idx;
        classification_idx = detect_idx;
        fetch_idx = (fetch_idx + 1) % 10;
*/        
/*
        if ( fetch_idx == 0 ) {
            since_now = 1;
        }
        if ( since_now ) {
            printf("thesis check 22 start fetch thread\n");
            //free_image(fetch_frame[fetch_idx].frame);
            //free_image(fetch_frame[fetch_idx].classi_frame);
        }
*/

        if (since_now) {
            printf("Thesis Check Before get_image_from_stream_resize_with_timestamp\n");
        //in_s = get_image_from_stream_resize_with_timestamp(cap, net.w, net.h, net.c, &in_img, dont_close_stream, &fetch_frame[fetch_idx], net_c[CLA].w, net_c[CLA].h);
            in_s = get_image_from_stream_resize_with_timestamp(cap, net[0].w, net[0].h, net[0].c, &in_img, dont_close_stream, &once_frame[0], net_c[CLA].w, net_c[CLA].h);
            
            //in_s = get_image_from_stream_resize_with_timestamp(cap, net[2].w, net[2].h, net[2].c, &in_img, dont_close_stream, &once_frame[2], net_c[CLA].w, net_c[CLA].h);
            
            printf("Thesis Success Capture Image in Fetch Thread\n");
            since_now = 0;
        }
        
        

        webcam_is_ok = 1;
//        show_img = in_img;
    
// unlock() 

        if (!in_s.data) {
            printf("\n\n\n\nStream closed.\n\n");
            custom_atomic_store_int(&flag_exit, 1);
            //exit(EXIT_FAILURE);
            return 0;
        }
//        release_mat(&in_img);
//        free_image(in_s);
       
        //in_s = resize_image(in, net.w, net.h);
    }
    printf("Thesis Out of Boundary Fetch Thread\n");

    return 0;
}

void *multi_detect_in_thread(void *ptr)
{
    printf("thesis check start detect thread\n");
    int det_thread_idx = *((int *)ptr);
    while (!custom_atomic_load_int(&flag_exit)) {
        printf("thesis check in detect thread while\n");
        while (!custom_atomic_load_int(&run_detect_in_thread[det_thread_idx])) {
            if (custom_atomic_load_int(&flag_exit)) return 0;
            this_thread_yield();
        }
        printf("Thesis Check Start Detect Thread\n");
        //dnn_buffer[DET].detect_section = fetch_frame[detect_idx];
        dnn_buffer[det_thread_idx].detect_section = once_frame[det_thread_idx];
        dnn_buffer[det_thread_idx].frame_timestamp[det_count] = dnn_buffer[det_thread_idx].detect_section.timestamp;
/*        
        det_s = in_s;
        det_img = in_img;
*/
        det_s = dnn_buffer[det_thread_idx].detect_section.frame;
//        det_img = (mat_cv*)dnn_buffer[DET].detect_section.img;


        int tmp_count = det_count;

//        layer l = net.layers[net.n - 1];
        float *X = det_s.data;
        //float *prediction = 
        dnn_buffer[det_thread_idx].before_prediction[tmp_count] = multi_get_wall_time();

        //network_predict(net, X);
//        nvtx_name_DET = nvtxRangeStartA("MultiDNN - Detector");
//        printf("Thesis Check Start network_predict in Detect Thread\n");
        //multi_network_predict(net[det_thread_idx], X, dnn_buffer[det_thread_idx].info);
        multi_network_predict(net[det_thread_idx], X, &dnn_buffer[det_thread_idx]);
//        printf("Thesis Check End network_predict in Detect Thread\n");
//        nvtxRangeEnd(nvtx_name_DET);

        dnn_buffer[det_thread_idx].after_prediction[tmp_count] = multi_get_wall_time();
        printf("Detector DET: %d, tmp_count: %d, prediction_time: %0.1lf\n",
        det_thread_idx, tmp_count, dnn_buffer[det_thread_idx].after_prediction[tmp_count]-dnn_buffer[det_thread_idx].before_prediction[tmp_count]);

/*
        cv_images[demo_index] = det_img;
        det_img = cv_images[(demo_index + avg_frames / 2 + 1) % avg_frames];
        demo_index = (demo_index + 1) % avg_frames;
*/        
  //      printf("Thesis Check 2 Detect Thread\n");
/*
        if (letter_box)
            dets = get_network_boxes(&net, get_width_mat(in_img), get_height_mat(in_img), demo_thresh, demo_thresh, 0, 1, &nboxes, 1); // letter box
        else
            dets = get_network_boxes(&net, net.w, net.h, demo_thresh, demo_thresh, 0, 1, &nboxes, 0); // resized
*/            
//        show_img = det_img;
//        printf("Thesis Check 3 Detect Thread\n");

        dnn_buffer[det_thread_idx].release = 1; // += 1?

        //const float nms = .45;
        //if (nms) {
        //    if (l.nms_kind == DEFAULT_NMS) do_nms_sort(dets, nboxes, l.classes, nms);
        //    else diounms_sort(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
        //}
//        printf("thesis detect, nboxes %d\n", nboxes);

        custom_atomic_store_int(&run_detect_in_thread[det_thread_idx], 0);
    }

    printf("Thesis Out of Boundary Detect Thread\n");
    return 0;
}

void *demo_detector_thread(void *arg)
{
    DemoDetector *argm = (DemoDetector *)arg;

    int idx = argm->idx;
    char *cfgfile = argm->cfgfile;
    char *weightfile = argm->weightfile;
    float thresh = argm->thresh;
    float hier_thresh = argm->hier_thresh;
//    int cam_index = argm->cam_index;
    int cam_index = 1;
    const char *filename = argm->filename;
    char **names = argm->names;
    int classes = argm->classes;
    int avgframes = argm->avgframes;
    int frame_skip = argm->frame_skip;
    char *prefix = argm->prefix;
    char *out_filename = out_filename;
    int mjpeg_port = argm->mjpeg_port;
    int dontdraw_bbox = argm->dontdraw_bbox;
    int json_port = argm->json_port;
    int dont_show = argm->dont_show;
    int ext_output = argm->ext_output;
    int letter_box_in = argm->letter_box;
    int time_limit_sec = argm->time_limit_sec;
    char *http_post_host = argm->http_post_host;
    int benchmark = argm->benchmark;
    int benchmark_layers = argm->benchmark_layers;


    if (avgframes < 1) avgframes = 1;
    avg_frames = avgframes;
    int letter_box = letter_box_in;
    in_img = det_img = show_img = NULL;
    //skip = frame_skip;
    image **alphabet = load_alphabet();
    int delay = frame_skip;
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
    demo_ext_output = ext_output;
    demo_json_port = json_port;
    printf("Demo\n");


    net[idx] = parse_network_cfg_custom(cfgfile, 1, 1);    // set batch=1
    if(weightfile){
        load_weights(&net[idx], weightfile);
    }

    printf("---- Success Detector Network setting\n");

    if (net[idx].letter_box) letter_box = 1;
    net[idx].benchmark_layers = benchmark_layers;
    fuse_conv_batchnorm(net[idx]);
    calculate_binary_weights(net[idx]);
    srand(2222222);

    dnn_buffer[idx].on = 1;

}


void run_multidnn(int argc, char **argv)
{
    int numberof_dnn = atoi(argv[2]);
    pthread_t period_thread[3];
    Args *args = (Args *)malloc(sizeof(Args));
    
    args->argc = argc;
    args->argv = argv;

#ifdef MULTIDNN     
    printf("Run Multi-DNN\n");
#endif
#ifdef MEASUREMENT
    printf("    Will Measure\n");
#endif
#ifdef BASIC_MULTIDNN
    printf("    B-A-S-I-C Scheduling\n");
#elif PRIORITY_MULTIDNN
    printf("    P-R-I-O-R-I-T-Y Scheduling\n");
#elif PREEMPTION_MULTIDNN
    printf("    P-R-E-E-M-P-T-I-O-N Scheduling\n");
#endif

    int cam_index = find_int_arg(argc, argv, "-c", 0);
    char *filename = (argc > 17) ? argv[17]: 0;
    if(filename){
        printf("video file: %s\n", filename);
        cap = get_capture_video_stream(filename);
    }else{
        printf("Webcam index: %d\n", cam_index);
        cap = get_capture_webcam(cam_index);
    }
    
    custom_thread_t fetch_thread = NULL;
    printf("thesis check before create fetch\n");
    if (custom_create_thread(&fetch_thread, 0, multi_fetch_in_thread, 0)) error("Thread creation failed", DARKNET_LOC);


    custom_thread_t classification_thread[10];
    
    for (int i = 0 ; i < numberof_dnn ; ++i) {
//    for (int i = 1 ; i >= 0; --i) {
        args->idx = i;
        if (i != 1) { // Detector: yolo
            printf("-------------------------------------------\n");
            printf("Detector Create Part\n");
            int idx = i;
         
            char *name=NULL;
            int prior = 0;
            struct timespec period;
            char *cfg;
            char *weights;
           
            if(0==strcmp(argv[6], "yolov2") && (i==0)) {
                name = "yolov2";
                period.tv_sec = YOLOV2_SEC;
                period.tv_nsec = YOLOV2_NSEC;
                prior = YOLO_PRIOR;
                cfg = argv[7];
//            char *weights = (argc > 8) ? argv[8] : 0;
                weights = argv[8];

            } else if(0==strcmp(argv[6], "yolov3") && (i==0)) {
                name = "yolov3";
                period.tv_sec = YOLOV3_SEC;
                period.tv_nsec = YOLOV3_NSEC;
                prior = YOLO_PRIOR;
                cfg = argv[7];
//            char *weights = (argc > 8) ? argv[8] : 0;
                weights = argv[8];

            } else if(0==strcmp(argv[5], "yolov4")) {
                name = "yolov4";
                period.tv_sec = YOLOV4_SEC;
                period.tv_nsec = YOLOV4_NSEC;
                prior = YOLO_PRIOR;
                cfg = argv[7];
                weights = argv[8];
        
            } else if(0==strcmp(argv[5], "yolov2-tiny")) {
                name = "yolov2-tiny";
                period.tv_sec = YOLOV2T_SEC;
                period.tv_nsec = YOLOV2T_NSEC;
                prior = YOLO_PRIOR;
        
            } else if(0==strcmp(argv[5], "yolov3-tiny")) {
                name = "yolov3-tiny";
                period.tv_sec = YOLOV3T_SEC;
                period.tv_nsec = YOLOV3T_NSEC;
                prior = YOLO_PRIOR;
        
            } else if(0==strcmp(argv[5], "yolov4-tiny")) {
                name = "yolov4-tiny";
                period.tv_sec = YOLOV4T_SEC;
                period.tv_nsec = YOLOV4T_NSEC;
                prior = YOLO_PRIOR;
        
            } else {
                perror("No match Network ");
                exit(0);
            }
        
        
            dnn_buffer[idx].on = 0;
            dnn_buffer[idx].period = period;
            dnn_buffer[idx].info.name = name;
            dnn_buffer[idx].info.ID = idx;
            dnn_buffer[idx].info.prior = prior;
            dnn_buffer[idx].info.stream_number = idx+1;
        
            printf("- This part create DNN: \n");
            printf("       Name: %s\n", dnn_buffer[idx].info.name);
            printf("     Period: %ld ms\n", dnn_buffer[idx].period.tv_nsec/1000000);
            printf("   Priority: %d\n", dnn_buffer[idx].info.prior);
            
            int dont_show = find_arg(argc, argv, "-dont_show");
            int benchmark = find_arg(argc, argv, "-benchmark");
            int benchmark_layers = find_arg(argc, argv, "-benchmark_layers");
            //if (benchmark_layers) benchmark = 1;
            if (benchmark) dont_show = 1;
            int letter_box = find_arg(argc, argv, "-letter_box");
            int mjpeg_port = find_int_arg(argc, argv, "-mjpeg_port", -1);
            int avgframes = find_int_arg(argc, argv, "-avgframes", 3);
            int dontdraw_bbox = find_arg(argc, argv, "-dontdraw_bbox");
            int json_port = find_int_arg(argc, argv, "-json_port", -1);
            char *http_post_host = find_char_arg(argc, argv, "-http_post_host", 0);
            int time_limit_sec = find_int_arg(argc, argv, "-time_limit_sec", 0);
            char *out_filename = find_char_arg(argc, argv, "-out_filename", 0);
            char *prefix = find_char_arg(argc, argv, "-prefix", 0);
            float thresh = find_float_arg(argc, argv, "-thresh", .25);    // 0.24
            float hier_thresh = find_float_arg(argc, argv, "-hier", .5);
            int cam_index = find_int_arg(argc, argv, "-c", 0);
            int frame_skip = find_int_arg(argc, argv, "-s", 0);
            int ext_output = find_arg(argc, argv, "-ext_output");
            
            char *datacfg = argv[3];
//            char *filename = (argc > 17) ? argv[17] : 0;
        
            list *options = read_data_cfg(datacfg);
            int classes = option_find_int(options, "classes", 20);
            char *name_list = option_find_str(options, "names", "data/names.list");
            char **names = get_labels(name_list);
        
            DemoDetector *args = (DemoDetector *)malloc(sizeof(DemoDetector));
            
            args->idx = idx;
            args->cfgfile = cfg;
            args->weightfile = weights;
            args->thresh = thresh;
            args->hier_thresh = hier_thresh;
            args->cam_index = cam_index;
            args->filename = filename;
            args->names = names;
            args->classes = classes;
            args->avgframes = avgframes;
            args->frame_skip = frame_skip;
            args->prefix = prefix;
            args->out_filename = out_filename;
            args->mjpeg_port = mjpeg_port;
            args->dontdraw_bbox = dontdraw_bbox;
            args->json_port = json_port;
            args->ext_output = ext_output;
            args->letter_box = letter_box;
            args->time_limit_sec = time_limit_sec;
            args->http_post_host = http_post_host;
            args->benchmark = benchmark;
            args->benchmark_layers = benchmark_layers;
        
            demo_detector_thread( (void *) args);
                    
            while (!dnn_buffer[idx].on) {};
            free(args);
   
            custom_thread_t detect_thread = NULL;
            if (custom_create_thread(&detect_thread, 0, multi_detect_in_thread, (void *)&idx)) error("Thread creation failed", DARKNET_LOC);
        
            printf("---------------------------- Creat Complete Detector\n");


        } else { // Classifier: alexnet, darknet19
            int idx = i; // 1 ~
            printf("-------------------------------------------\n");
            printf("Classification Create Part %d\n", idx);
        
            char *name=NULL;
            int prior = 0;
            struct timespec period;
            char *data = argv[4];

            char *cfg;
            char *weights;
        
            if(0==strcmp(argv[6], "alexnet") && (i==1)) {
                name = "alexnet";
                period.tv_sec = ALEXNET_SEC;
                period.tv_nsec = ALEXNET_NSEC;
                prior = DENSENET201_PRIOR;
                cfg = argv[9];
                weights = argv[10];

                
        
            } else if(0==strcmp(argv[7], "darknet19") && (i<=2)) {
                name = "darknet19";
                period.tv_sec = DARKNET19_SEC;
                period.tv_nsec = DARKNET19_NSEC;
                prior = DARKNET19_PRIOR;
                cfg = argv[13];
                weights = argv[14];

            } else if(0==strcmp(argv[7], "darknet53") && (i<=2)) {
                //name = "alexnet";
                name = "darknet53";
                period.tv_sec = ALEXNET_SEC;
                period.tv_nsec = ALEXNET_NSEC;
                prior = ALEXNET_PRIOR;
                cfg = argv[12];
                weights = argv[13];

            } else {
                perror("No match Network ");
                exit(0);
            }
        
            dnn_buffer[idx].on = 0;
            dnn_buffer[idx].period = period;
            dnn_buffer[idx].info.name = name;
            dnn_buffer[idx].info.ID = idx;
            dnn_buffer[idx].info.prior = prior;
            dnn_buffer[idx].info.stream_number = idx+1;
        
            printf("- This part create DNN: \n");
            printf("       Name: %s\n", dnn_buffer[idx].info.name);
            printf("     Period: %ld ms\n", dnn_buffer[idx].period.tv_nsec/1000000);
            printf("   Priority: %d\n", dnn_buffer[idx].info.prior);
            
        
        
            int benchmark = find_arg(argc, argv, "-benchmark");
            int benchmark_layers = find_arg(argc, argv, "-benchmark_layers");
            if (benchmark_layers) benchmark = 1;
            int cam_index = find_int_arg(argc, argv, "-c", 0);
                    
            DemoClassi *args = (DemoClassi *)malloc(sizeof(DemoClassi));
            
            args->idx = idx;
            args->datacfg = data;
            args->cfgfile = cfg;
            args->weightfile = weights;
            args->cam_index = cam_index;
            args->filename = filename;
            args->benchmark = benchmark;
            args->benchmark_layers = benchmark_layers;
        
            demo_classification_thread( (void *) args);
                    
            while (!dnn_buffer[idx].on) {};
            free(args);
    
            int tmp_integer = args->idx;
            if (custom_create_thread(&classification_thread[idx], NULL, multi_classification_in_thread, (void *)&idx)) error("Thread creation failed", DARKNET_LOC);
        
            printf("---------------------------- Creat Complete\n");
        } // if else end
        sleep(1);

    } // for end

    


    int start_flag = 1;
    printf("\n\n\n\n\n");

    while(!dnn_buffer[0].on) {};
// mutex main

    struct timespec period;
    period.tv_sec = QUANTUM_SEC;
    period.tv_nsec = QUANTUM_NSEC;

    double detector_timer = 0;
    double classifier_timer = 0;
    double tmp_timer = 0;

    int meas_counter = 0;
    int meas = 0;
    double present_quantum_time = 0;
    int classifier_count = 0;
    int detector_count = 0;
    double start_quantum = 0;
    
    while(1) {
        

//        printf("\n\n\nThesis Check Start   Main Quantum Counter\n\n\n");
//        printf("%d\n", dnn_buffer[0].on);
//        printf("%d\n", dnn_buffer[numberof_dnn-1].on);
//        printf("%d\n", webcam_is_ok);
        if(dnn_buffer[0].on && dnn_buffer[numberof_dnn-1].on && webcam_is_ok) {
//        if(dnn_buffer[0].on && webcam_is_ok) {

//            printf("\n\n\nThesis Check Running Main Quantum Counter\n\n\n");
//            printf("Thesis meas_counter: %d\n", meas_counter);
            start_quantum = multi_get_wall_time();
            detector_timer += (present_quantum_time);
            classifier_timer += (present_quantum_time);
            tmp_timer += (present_quantum_time);

#ifdef PRIORITY_MULTIDNN
            
            if (classifier_timer >= CLASSIFIER_PERIOD) {
                printf("\n\n\nClassifier Release, Time: %.4lf\n\n\n", multi_get_wall_time());

                dnn_buffer[CLA].count = classifier_count;
                cla_count = classifier_count;
                
                custom_atomic_store_int(&run_classification_in_thread[1], 1);
                
                dnn_buffer[CLA].release_period[classifier_count] = classifier_timer;
                dnn_buffer[CLA].release_time[classifier_count] = multi_get_wall_time();
//                printf("Classifier count: %d, release: %.4lf\n", classifier_count, dnn_buffer[CLA].release_time[classifier_count]);

                classifier_timer = 0;
                classifier_count += 1;
            }
            double tmp_timer = multi_get_wall_time() - start_quantum;
#endif            

            double c_timer = multi_get_wall_time() - start_quantum;
            if (tmp_timer + c_timer >= TMP_PERIOD) {
                printf("\n\n\nDetector Release, Time: %.1lf\n\n\n", multi_get_wall_time());


                custom_atomic_store_int(&run_detect_in_thread[0], 1);

                detector_count += 1;
                tmp_timer = 0;   
            }

//                // Alexnet
//            double c_timer = multi_get_wall_time() - start_quantum;
//            if (tmp_timer + c_timer >= TMP_PERIOD) {
////            if (classifier_timer >= CLASSIFIER_PERIOD) {
//                printf("\n\n\nClassifier Release, Time: %.4lf\n\n\n", multi_get_wall_time());
//
//
//                custom_atomic_store_int(&run_classification_in_thread[2], 1);
//                
//                dnn_buffer[CLA].release_period[classifier_count] = classifier_timer + tmp_timer;
//                dnn_buffer[CLA].release_time[classifier_count] = multi_get_wall_time();
////                printf("Classifier count: %d, release: %.4lf\n", classifier_count, dnn_buffer[CLA].release_time[classifier_count]);
//
//                tmp_timer = 0;
//                tmp_c += 1;
//            }

            if (detector_timer >= DETECTOR_PERIOD) {
                printf("\n\n\nDetector Release, Time: %.1lf\n\n\n", multi_get_wall_time());

                dnn_buffer[DET].count = detector_count;
                det_count = detector_count;

                custom_atomic_store_int(&run_detect_in_thread[2], 1);
#ifdef PRIORITY_MULTIDNN
                dnn_buffer[DET].release_period[detector_count] = detector_timer + tmp_timer;
#endif
#ifndef PRIORITY_MULTIDNN                
                dnn_buffer[DET].release_period[detector_count] = detector_timer;
#endif                
                dnn_buffer[DET].release_time[detector_count] = multi_get_wall_time();

                detector_count += 1;
                detector_timer = 0;   
            }
#ifndef PRIORITY_MULTIDNN
            double tmp_timer = multi_get_wall_time() - start_quantum;
            if (classifier_timer + tmp_timer >= CLASSIFIER_PERIOD) {
//            if (classifier_timer >= CLASSIFIER_PERIOD) {
                printf("\n\n\nClassifier Release, Time: %.4lf\n\n\n", multi_get_wall_time());
                
                dnn_buffer[CLA].count = classifier_count;

                cla_count = classifier_count;

                custom_atomic_store_int(&run_classification_in_thread[1], 1);
                
                dnn_buffer[CLA].release_period[classifier_count] = classifier_timer + tmp_timer;
                dnn_buffer[CLA].release_time[classifier_count] = multi_get_wall_time();
//                printf("Classifier count: %d, release: %.4lf\n", classifier_count, dnn_buffer[CLA].release_time[classifier_count]);

                classifier_timer = 0;
                classifier_count += 1;
            }
#endif

#ifdef MEASUREMENT
//            meas = dnn_buffer[0].count - MEAS_THRESHOLD;
//            meas = dnn_buffer[1].count;
            meas = classifier_count - MEAS_THRESHOLD;
            //meas = detector_count - MEAS_THRESHOLD;
/*            
            if (meas >= 0) {
                meas_quantum_time_array[meas] = present_quantum_time;


            }
*/
            if (meas >= MEASUREMENT_ITERATION - MEAS_THRESHOLD - 1) {
                printf(" Thesis Start Recording Measure, Detector\n");
                
                int exist = 0;
                FILE *fp;
                char file_path_d[100] = "";
                char file_path_c[100] = "";

                strcat(file_path_d, MEASUREMENT_PATH);
                strcat(file_path_d, MEASUREMENTD_FILE);
/*
                fp = fopen(file_path_d, "w+");

                if (fp == NULL) {
                    int result;

                    result = mkdir(MEASUREMENT_PATH, 0766);

                    if(result == 0) {
                        exist = 1;
                        fp = fopen(file_path_d, "w+");
                    }
                }

                fprintf(fp, "%s\n", "D_DetecTime");

                for (int i = MEAS_THRESHOLD; i < dnn_buffer[DET].count; ++i) {
                    fprintf(fp, "%.1lf\n", 
                    dnn_buffer[DET].after_prediction[i] - dnn_buffer[DET].before_prediction[i]);
                }

                fclose(fp);*/
                printf(" Thesis Complete Recording Measure, Detector\n");

                
                printf(" Thesis Start Recording Measure, Classifier\n");
                strcat(file_path_c, MEASUREMENT_PATH);
                strcat(file_path_c, MEASUREMENTC_FILE);

                fp = fopen(file_path_c, "w+");

                if (fp == NULL) {
                    int result;

                    result = mkdir(MEASUREMENT_PATH, 0766);

                    if(result == 0) {
                        exist = 1;
                        fp = fopen(file_path_c, "w+");
                    }
                }

                fprintf(fp, "%s,%s,%s\n", "d201", "a", "over");

                for (int i = MEAS_THRESHOLD; i < dnn_buffer[CLA].count; ++i) {
                    fprintf(fp, "%.1lf,%.1lf,%.2lf\n", 
                    dnn_buffer[1].prediction_time[i], dnn_buffer[2].prediction_time[i], dnn_buffer[1].overhead[i]);
                }

                fclose(fp);
                printf(" Thesis Complete Recording Measure, Classifier\n");

                system_exit_flag = 1;
                break;
            }

            meas_counter += 1;
#endif

#ifdef QUANTUM_MEASUREMENT
            meas = meas_counter - MEAS_THRESHOLD;
            if (meas >= 0) {
                meas_quantum_time_array[meas] = present_quantum_time;


            }

            if (meas >= QUANTUM_ITERATION - 1) {
                int exist = 0;
                FILE *fp;
                char file_path[100] = "";

                strcat(file_path, MEASUREMENT_PATH);
                strcat(file_path, MEASUREMENT_FILE);

                fp = fopen(file_path, "w+");

                if (fp == NULL) {
                    int result;

                    result = mkdir(MEASUREMENT_PATH, 0766);

                    if(result == 0) {
                        exist = 1;
                        fp = fopen(file_path, "w+");
                    }
                }

                fprintf(fp, "%s\n", "quantum_time");

                for (int i = 0; i < QUANTUM_ITERATION-1; ++i) {
                    fprintf(fp, "%.1lf\n", meas_quantum_time_array[i]*1000);
                }

                fclose(fp);


                break;
            }

            meas_counter += 1;
#endif
            nanosleep(&period, NULL);
            present_quantum_time = multi_get_wall_time() - start_quantum;
//            present_quantum_time = get_time_in_ms() - start_quantum;

            
        }


    }
}

