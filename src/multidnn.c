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


struct timespec thread_sleep = {THREAD_SLEEP_SEC, THREAD_SLEEP_NSEC};
//thread_sleep.tv_sec = THREAD_SLEEP_SEC;
//thread_sleep.tv_nsec = THREAD_SLEEP_NSEC;



#ifdef QUANTUM_MEASUREMENT

double meas_quantum_time_array[QUANTUM_ITERATION];

#endif

volatile system_exit_flag;

volatile int detector_display_flag;
volatile int classification_display_flag;

volatile int webcam_is_ok;

volatile int det_count = 0;
volatile int cla_count = 0;

ImageFrame fetch_frame[3];
int fetch_idx;
int detect_idx;
int classification_idx;


pthread_mutex_t main_lock = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t tmp_lock = PTHREAD_MUTEX_INITIALIZER;

volatile MultiDNN dnn_buffer[3];
pthread_t demo_thread[2];

/*****  DemoDetector *****/
static char **demo_names;
static image **demo_alphabet;
static int demo_classes;

static int nboxes = 0;
static detection *dets = NULL;

static network net;
static network net_c;
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
static volatile int run_detect_in_thread = 0;
static volatile int run_display_in_thread = 0;
static volatile int run_classification_in_thread = 0;

static volatile int run_prediction_gpu = 0;
/*************************/

double multi_get_wall_time()
{
/* 
    struct timeval walltime;
    
    if (gettimeofday(&walltime, NULL)) {
        return 0;
    }
    int tmp_sec = (int)walltime.tv_sec % 10000;
    int tmp_usec = (int)walltime.tv_usec;

    return (double)tmp_sec + (double)tmp_usec * .000001;
*/    
    
    struct timespec after_boot;
    clock_gettime(CLOCK_MONOTONIC, &after_boot);
    return (after_boot.tv_sec*1000 + after_boot.tv_nsec*0.000001);
    
}

double wall_time_1()
{
    struct timespec time_after_boot;
    clock_gettime(CLOCK_MONOTONIC, &time_after_boot);
    return ((double)time_after_boot.tv_sec*1000 + (double)time_after_boot.tv_nsec*0.000001);
}
double wall_time_2()
{
    struct timespec time_after_boot;
    clock_gettime(CLOCK_MONOTONIC, &time_after_boot);
    return ((double)time_after_boot.tv_sec*1000 + (double)time_after_boot.tv_nsec*0.000001);
}
double wall_time_3()
{
    struct timespec time_after_boot;
    clock_gettime(CLOCK_MONOTONIC, &time_after_boot);
    return ((double)time_after_boot.tv_sec*1000 + (double)time_after_boot.tv_nsec*0.000001);
}
double wall_time_4()
{
    struct timespec time_after_boot;
    clock_gettime(CLOCK_MONOTONIC, &time_after_boot);
    return ((double)time_after_boot.tv_sec*1000 + (double)time_after_boot.tv_nsec*0.000001);
}
double wall_time_5()
{
    struct timespec time_after_boot;
    clock_gettime(CLOCK_MONOTONIC, &time_after_boot);
    return ((double)time_after_boot.tv_sec*1000 + (double)time_after_boot.tv_nsec*0.000001);
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
    net_c = parse_network_cfg_custom(cfgfile, 1, 0);
    if(weightfile){
        load_weights(&net_c, weightfile);
    }

    printf("---- Success Classifier Network setting\n");

    net_c.benchmark_layers = benchmark_layers;
    set_batch_network(&net_c, 1);
    list *options = read_data_cfg(datacfg);

    fuse_conv_batchnorm(net_c);
    calculate_binary_weights(net_c);

    srand(2222222);
/*
    cap_cv * cap_c;

    if(filename){
        cap_c = get_capture_video_stream(filename);
    }else{
        cap_c = get_capture_webcam(cam_index);
    }
*/
    int classes = option_find_int(options, "classes", 2);
    int top = option_find_int(options, "top", 1);
    if (top > classes) top = classes;

    char *name_list = option_find_str(options, "names", 0);
    char **names = get_labels(name_list);

    int* indexes = (int*)xcalloc(top, sizeof(int));

//    if(!cap_c) error("Couldn't connect to webcam.", DARKNET_LOC);
//    if (!benchmark) create_window_cv("Classifier", 0, 512, 512);
//    if (!benchmark) create_window_cv("Demo", 0, 512, 512);
    float fps = 0;
    int i;

    double start_time = get_time_point();
    float avg_fps = 0;
    int frame_counter = 0;
    
// thesis
    double thesis_cycletime = 0;
    double thesis_executiontime = 0;
    double old_time = 0;
    double time = 0;
    double cycle_time = 0;
    int thesis_idx = 0;

    dnn_buffer[idx].on = 1;

    int tmp_count = 0;

    while(1){
        if(dnn_buffer[idx].release) {
            dnn_buffer[idx].release = 0;
            
            tmp_count += 1;
#ifdef MEASUREMENT
            
            double start_classifier = multi_get_wall_time();
            //dnn_buffer[idx].count = tmp_count - 1;
            dnn_buffer[idx].onrunning_time[cla_count] = start_classifier;

#endif
/*
            dnn_buffer[CLA].detect_section = fetch_frame[detect_idx];
            dnn_buffer[CLA].frame_timestamp[cla_count] = dnn_buffer[CLA].detect_section.timestamp;

            printf("Classifier On and On and On\n");
        
            struct timeval tval_before, tval_after, tval_result;
            gettimeofday(&tval_before, NULL);
    
            //image in = get_image_from_stream(cap);
            image in_c, in;
            in_c = resize_image(dnn_buffer[CLA].detect_section.classi_frame, net_c.w, net_c.h);
            dnn_buffer[CLA].detect_section.classi_frame = in_c;


            old_time = time;
            time = get_time_point();
            cycle_time = (time - old_time)/1000;

            dnn_buffer[CLA].before_prediction[cla_count] = multi_get_wall_time();

            float *predictions = network_predict(net_c, in_c.data);

            dnn_buffer[CLA].after_prediction[cla_count] = multi_get_wall_time();
*/




            float *predictions = dnn_buffer[CLA].prediction;
            double frame_time_ms = (get_time_point() - time)/1000;
   
            frame_counter++;
    // thesis
            thesis_idx += 1;      
    
            if(net.hierarchy) hierarchy_predictions(predictions, net_c.outputs, net_c.hierarchy, 1);
            top_predictions(net_c, top, indexes);
    
    #ifndef _WIN32
            printf("\033[2J");
            printf("\033[1;1H");
    #endif
    
    
            if (!benchmark) {
                printf("---------- Classifier Result ----------\n");
                printf("\ridx: %d, FPS: %.2f  (use -benchmark command line flag for correct measurement)\n", cla_count, fps);
                for (i = 0; i < top; ++i) {
                    int index = indexes[i];
                    printf("%.1f%%: %s\n", predictions[index] * 100, names[index]);
                }
                printf("---------- --- Classifier -- ----------\n");
                printf("\n");
    
//                free_image(in_c);
//                free_image(in);
    
//                printf("classifi check key 1\n");
//                int c = wait_key_cv(10);// cvWaitKey(10);
//                printf("classifi check key 2\n");
//                if (c == 27 || c == 1048603) break;
            }
            else {
                printf("\rFPS: %.2f \t AVG_FPS = %.2f ", fps, avg_fps);
            }
    
            //gettimeofday(&tval_after, NULL);
            //timersub(&tval_after, &tval_before, &tval_result);
            //float curr = 1000000.f/((long int)tval_result.tv_usec);
            float curr = 1000.f / frame_time_ms;
            if (fps == 0) fps = curr;
            else fps = .9*fps + .1*curr;
    
            float spent_time = (get_time_point() - start_time) / 1000000;
    
    // thesis  
    /*
            if(thesis_idx>25) {
                thesis_cycletime += cycle_time;
                thesis_executiontime += frame_time_ms;
            }
            if(thesis_idx==1025) {
                printf("Classifier: %s  \n", weightfile);
                printf("    Avg     Cycle time: %lf\n", thesis_cycletime/1000);
                printf("    Avg Execution time: %lf\n", thesis_executiontime/1000);
                break;
            }
    */
            if (spent_time >= 3.0f) {
                //printf(" spent_time = %f \n", spent_time);
                avg_fps = frame_counter / spent_time;
                frame_counter = 0;
                start_time = get_time_point();
            }
            
#ifdef MEASUREMENT
            
            double end_classifier = multi_get_wall_time();
//            double end_classifier = get_time_in_ms();
            dnn_buffer[idx].complete_time[cla_count] = end_classifier;
            
/*            
            if( meas >= 0 ) {
                double end_classifier = multi_get_wall_time();
                dnn_buffer[idx].complete_time[meas] = end_classifier;
            }
            if(system_exit_flag) break;
*/            
#endif

        }
    }

#endif
}

void *multi_classification_in_thread(void *ptr)
{
    printf("thesis check start classification thread\n");
    while (!custom_atomic_load_int(&flag_exit)) {
//        printf("thesis check in detect thread while\n");
        while (!custom_atomic_load_int(&run_classification_in_thread)) {
            if (custom_atomic_load_int(&flag_exit)) return 0;
            this_thread_yield();
        }
//        printf("Thesis Check Start Detect Thread\n");
        dnn_buffer[CLA].detect_section = fetch_frame[classification_idx];
        dnn_buffer[CLA].frame_timestamp[cla_count] = dnn_buffer[CLA].detect_section.timestamp;
/*        
        det_s = in_s;
        det_img = in_img;
*/


        image in_c, in;
        in_c = dnn_buffer[CLA].detect_section.classi_frame;
/*
        dnn_buffer[CLA].detect_section.classi_frame = in_c;
*/
        int tmp_count = cla_count;

        usleep(2000);
        dnn_buffer[CLA].before_prediction[tmp_count] = multi_get_wall_time();

        float *predictions = network_predict(net_c, in_c.data);

        dnn_buffer[CLA].after_prediction[tmp_count] = multi_get_wall_time();

        dnn_buffer[CLA].prediction = predictions;


        dnn_buffer[CLA].release = 1; // += 1?


        custom_atomic_store_int(&run_classification_in_thread, 0);
    }

    printf("Thesis Out of Boundary Detect Thread\n");
    return 0;
}

void *multi_fetch_in_thread(void *ptr)
{
    printf("thesis check start fetch thread\n");
    printf("Thesis Check dnn_buffer[0].on : %d\n", dnn_buffer[0].on);
    while (!custom_atomic_load_int(&flag_exit)) {
        while(!dnn_buffer[0].on) {};
//        printf("thesis check in fetch thread while\n");
        
        int dont_close_stream = 0;    // set 1 if your IP-camera periodically turns off and turns on video-stream

// lock(), in_s : fetch image data
   
/* 
        if (letter_box)
            in_s = get_image_from_stream_letterbox(cap, net.w, net.h, net.c, &in_img, dont_close_stream);
        else
            in_s = get_image_from_stream_resize(cap, net.w, net.h, net.c, &in_img, dont_close_stream);
*/
        in_s = get_image_from_stream_resize_with_timestamp(cap, net.w, net.h, net.c, &in_img, dont_close_stream, &fetch_frame[fetch_idx%3], net_c.w, net_c.h);
        detect_idx = fetch_idx % 3;
        classification_idx = detect_idx;
        fetch_idx += 1;

        webcam_is_ok = 1;
//        show_img = in_img;
    
//        printf("Thesis Success Capture Image in Fetch Thread\n");
// unlock() 
        if (!in_s.data) {
            printf("Stream closed.\n");
            custom_atomic_store_int(&flag_exit, 1);
            //exit(EXIT_FAILURE);
            return 0;
        }
        
        //in_s = resize_image(in, net.w, net.h);
    }
    printf("Thesis Out of Boundary Fetch Thread\n");

    return 0;
}

void *multi_detect_in_thread(void *ptr)
{
    printf("thesis check start detect thread\n");
    while (!custom_atomic_load_int(&flag_exit)) {
//        printf("thesis check in detect thread while\n");
        while (!custom_atomic_load_int(&run_detect_in_thread)) {
            if (custom_atomic_load_int(&flag_exit)) return 0;
            this_thread_yield();
        }
//        printf("Thesis Check Start Detect Thread\n");
        dnn_buffer[DET].detect_section = fetch_frame[detect_idx];
        dnn_buffer[DET].frame_timestamp[det_count] = dnn_buffer[DET].detect_section.timestamp;
/*        
        det_s = in_s;
        det_img = in_img;
*/
        det_s = dnn_buffer[DET].detect_section.frame;
        det_img = (mat_cv*)dnn_buffer[DET].detect_section.img;


        int tmp_count = det_count;

//        layer l = net.layers[net.n - 1];
        float *X = det_s.data;
        //float *prediction = 
        dnn_buffer[DET].before_prediction[tmp_count] = multi_get_wall_time();
//        dnn_buffer[DET].before_prediction[tmp_count] = get_time_in_ms();

        network_predict(net, X);

        dnn_buffer[DET].after_prediction[tmp_count] = multi_get_wall_time();
//        dnn_buffer[DET].after_prediction[tmp_count] = get_time_in_ms();

        cv_images[demo_index] = det_img;
        det_img = cv_images[(demo_index + avg_frames / 2 + 1) % avg_frames];
        demo_index = (demo_index + 1) % avg_frames;
//        printf("hansaem detect, demo_thresh %f\n", demo_thresh);

        if (letter_box)
            dets = get_network_boxes(&net, get_width_mat(in_img), get_height_mat(in_img), demo_thresh, demo_thresh, 0, 1, &nboxes, 1); // letter box
        else
            dets = get_network_boxes(&net, net.w, net.h, demo_thresh, demo_thresh, 0, 1, &nboxes, 0); // resized
//        show_img = det_img;
        dnn_buffer[DET].display_section = dnn_buffer[DET].detect_section;
        dnn_buffer[DET].display_section.img = (void *)det_img;

        dnn_buffer[DET].release = 1; // += 1?

        //const float nms = .45;
        //if (nms) {
        //    if (l.nms_kind == DEFAULT_NMS) do_nms_sort(dets, nboxes, l.classes, nms);
        //    else diounms_sort(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
        //}
//        printf("thesis detect, nboxes %d\n", nboxes);

        custom_atomic_store_int(&run_detect_in_thread, 0);
    }

    printf("Thesis Out of Boundary Detect Thread\n");
    return 0;
}

void *multi_display_in_thread_sync(void *ptr)
{
    custom_atomic_store_int(&run_display_in_thread, 1);
    while (custom_atomic_load_int(&run_display_in_thread)) this_thread_sleep_for(thread_wait_ms);
    return 0;
}

void *multi_display_in_thread(void *arg)
{
    create_window_cv("MultiDNN", 0, 512, 512);
    while (!custom_atomic_load_int(&flag_exit)) {
        while (!custom_atomic_load_int(&run_display_in_thread)) {
            if (custom_atomic_load_int(&flag_exit)) return 0;
            this_thread_yield();
        }

//        printf("Thesis Check in Display Running\n");
        dnn_buffer[DET].display_start[det_count] = multi_get_wall_time();
//        dnn_buffer[DET].display_start[det_count] = get_time_in_ms();

        if(detector_display_flag) {
            show_image_mat(show_img, "MultiDNN");
            int c = wait_key_cv(1);
        } else if(classification_display_flag) {

        }

        dnn_buffer[DET].display_end[det_count] = multi_get_wall_time();
//        dnn_buffer[DET].display_end[det_count] = get_time_in_ms();

        custom_atomic_store_int(&run_display_in_thread, 0);

    }

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


    net = parse_network_cfg_custom(cfgfile, 1, 1);    // set batch=1
    if(weightfile){
        load_weights(&net, weightfile);
    }

    printf("---- Success Detector Network setting\n");

    if (net.letter_box) letter_box = 1;
    net.benchmark_layers = benchmark_layers;
    fuse_conv_batchnorm(net);
    calculate_binary_weights(net);
    srand(2222222);

    printf("thesis check 1\n");

    if(filename){
        printf("video file: %s\n", filename);
//        cap = get_capture_video_stream(filename);
        demo_skip_frame = is_live_stream(filename);
    }else{
        printf("Webcam index: %d\n", cam_index);
//        cap = get_capture_webcam(cam_index);
        printf("thesis check 1-1\n");
        demo_skip_frame = true;
    }

    printf("thesis check 2\n");
    if (!cap) {
#ifdef WIN32
        printf("Check that you have copied file opencv_ffmpeg340_64.dll to the same directory where is darknet.exe \n");
#endif
        error("Couldn't connect to webcam.", DARKNET_LOC);
    }

    layer l = net.layers[net.n-1];
    int j;

    cv_images = (mat_cv**)xcalloc(avg_frames, sizeof(mat_cv));

    printf("thesis check 3\n");
    int i;
    for (i = 0; i < net.n; ++i) {
        layer lc = net.layers[i];
        if (lc.type == YOLO) {
            lc.mean_alpha = 1.0 / avg_frames;
            l = lc;
        }
    }

    printf("thesis check 4\n");
    if (l.classes != demo_classes) {
        printf("\n Parameters don't match: in cfg-file classes=%d, in data-file classes=%d \n", l.classes, demo_classes);
        getchar();
        exit(0);
    }

    printf("thesis check 5\n");
    flag_exit = 0;
    int count = 0;
    if(!prefix && !dont_show){
        int full_screen = 0;
        printf("thesis check 9-1\n");
//        create_window_cv("Demo", full_screen, 1352, 1013);
//        create_window_cv("MultiDNN", 0, 512, 512);
        printf("thesis check 9-2\n");
    }


    printf("thesis check 10\n");
    write_cv* output_video_writer = NULL;
    if (out_filename && !flag_exit)
    {
        int src_fps = 25;
        src_fps = get_stream_fps_cpp_cv(cap);
        output_video_writer =
            create_video_writer(out_filename, 'D', 'I', 'V', 'X', src_fps, get_width_mat(det_img), get_height_mat(det_img), 1);

        //'H', '2', '6', '4'
        //'D', 'I', 'V', 'X'
        //'M', 'J', 'P', 'G'
        //'M', 'P', '4', 'V'
        //'M', 'P', '4', '2'
        //'X', 'V', 'I', 'D'
        //'W', 'M', 'V', '2'
    }

    int send_http_post_once = 0;
    const double start_time_lim = get_time_point();
    double before = get_time_point();
    double start_time = get_time_point();
    float avg_fps = 0;
    int frame_counter = 0;
    int global_frame_counter = 0;

// mutex main
    pthread_mutex_lock(&main_lock);

// thesis
    printf("THESIS hansaem demo.c net size : %ld\n", sizeof(net));
    dnn_buffer[idx].on = 1;

    int tmp_count = 0;

    while(1){
        ++count;
        if(dnn_buffer[idx].release) {
            tmp_count += 1;
#ifdef MEASUREMENT
            
            double start_detector = multi_get_wall_time();
//            double start_detector = get_time_in_ms();
//            detector_count = dnn_buffer[idx].count;
//            dnn_buffer[idx].count = tmp_count - 1;
            dnn_buffer[idx].onrunning_time[det_count] = start_detector;
                        
/*            
            int meas = tmp_count - MEAS_THRESHOLD; 
            if( meas >= 0 ) {
                double start_detector = multi_get_wall_time();
                dnn_buffer[idx].count = meas;
                dnn_buffer[idx].onrunning_time[meas] = start_detector;
            }
*/            
#endif
            dnn_buffer[idx].release = 0;
            printf("Multi Dnn Check Onetwo Onetwo\n");
            
            {
                const float nms = .45;    // 0.4F
                int local_nboxes = nboxes;
                detection *local_dets = dets;
                this_thread_yield();
                if (nms) {
                    if (l.nms_kind == DEFAULT_NMS) do_nms_sort(local_dets, local_nboxes, l.classes, nms);
                    else diounms_sort(local_dets, local_nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
                }
    
                if (l.embedding_size) set_track_id(local_dets, local_nboxes, demo_thresh, l.sim_thresh, l.track_ciou_norm, l.track_history_size, l.dets_for_track, l.dets_for_show);
    
                printf("----------- Detector Result %d -----------\n", det_count);
                show_img = (mat_cv*)dnn_buffer[DET].display_section.img;
                if (!benchmark && !dontdraw_bbox) draw_detections_cv_v3(show_img, local_dets, local_nboxes, demo_thresh, demo_names, demo_alphabet, demo_classes, demo_ext_output);
                free_detections(local_dets, local_nboxes);
    
                printf("\nFPS:%.1f \t AVG_FPS:%.1f\n", fps, avg_fps);
                printf("----------- -- Detector -- -----------\n");
    
                if(!prefix){
                    if (!dont_show) {
                        const int each_frame = max_val_cmp(1, avg_fps / 60);
                        
                        //dnn_buffer[DET].frame_timestamp[det_count] = dnn_buffer[DET].display_section.timestamp;
                        custom_atomic_store_int(&run_display_in_thread, 1);
                        detector_display_flag = 1;
                    }
                }else{
                    char buff[256];
                    sprintf(buff, "%s_%08d.jpg", prefix, count);
                    if(show_img) save_cv_jpg(show_img, buff);
                }

                while (custom_atomic_load_int(&run_display_in_thread)) {
                    if(avg_fps > 180) this_thread_yield();
                    else this_thread_sleep_for(thread_wait_ms);   // custom_join(detect_thread, 0);
                }

    
                if(delay == 0){
                    if(!benchmark) release_mat(&show_img);
                }
            }
            --delay;
            if(delay < 0){
                delay = frame_skip;
    
                //double after = get_wall_time();
                //float curr = 1./(after - before);
                double after = get_time_point();    // more accurate time measurements
                float curr = 1000000. / (after - before);
                fps = fps*0.9 + curr*0.1;
                before = after;
    
                float spent_time = (get_time_point() - start_time) / 1000000;
                frame_counter++;
                global_frame_counter++;
                if (spent_time >= 3.0f) {
                    //printf(" spent_time = %f \n", spent_time);
                    avg_fps = frame_counter / spent_time;
                    frame_counter = 0;
                    start_time = get_time_point();
                }
            }

#ifdef MEASUREMENT

            double end_detector = multi_get_wall_time();
//            double end_detector = get_time_in_ms();
            dnn_buffer[idx].complete_time[det_count] = end_detector;
            
/*
            if( meas >= 0 ) {
                double end_detector = multi_get_wall_time();
                dnn_buffer[idx].complete_time[meas] = end_detector;
            }
*/          
            if(system_exit_flag) break;
#endif
            dnn_buffer[idx].release = 0;

        }    

    }
    pthread_mutex_unlock(&main_lock);
    printf("input video stream closed. \n");
    if (output_video_writer) {
        release_video_writer(&output_video_writer);
        printf("output_video_writer closed. \n");
    }

    this_thread_sleep_for(thread_wait_ms);

//    custom_join(detect_thread, 0);
//    custom_join(fetch_thread, 0);

    // free memory
    free_image(in_s);
    free_detections(dets, nboxes);

    demo_index = (avg_frames + demo_index - 1) % avg_frames;
    for (j = 0; j < avg_frames; ++j) {
            release_mat(&cv_images[j]);
    }
    free(cv_images);

    free_ptrs((void **)names, net.layers[net.n - 1].classes);

    const int nsize = 8;
    for (j = 0; j < nsize; ++j) {
        for (i = 32; i < 127; ++i) {
            free_image(alphabet[j][i]);
        }
        free(alphabet[j]);
    }
    free(alphabet);
    free_network(net);
    //cudaProfilerStop();
}


void run_multidnn(int argc, char **argv)
{
    int numberof_dnn = atoi(argv[2]);
    pthread_t period_thread[3];
    Args *args = (Args *)malloc(sizeof(Args));
    
    args->argc = argc;
    args->argv = argv;

    printf("Run Multi-DNN\n");

    int cam_index = find_int_arg(argc, argv, "-c", 0);
    char *filename = (argc > 11) ? argv[11]: 0;
    if(filename){
        printf("video file: %s\n", filename);
        cap = get_capture_video_stream(filename);
    }else{
        printf("Webcam index: %d\n", cam_index);
        cap = get_capture_webcam(cam_index);
    }

    
    //for (int i = 0 ; i < numberof_dnn ; ++i) {
    for (int i = 1 ; i >= 0; --i) {
        args->idx = i;
        if (i == 0) { // Detector: yolo
            printf("-------------------------------------------\n");
            printf("Detector Create Part\n");
            int idx = i;
         
            char *name=NULL;
            int prior = 0;
            struct timespec period;
           
            if(0==strcmp(argv[3+idx], "yolov2")) {
                name = "yolov2";
                period.tv_sec = YOLOV2_SEC;
                period.tv_nsec = YOLOV2_NSEC;
                prior = YOLO_PRIOR;
        
            } else if(0==strcmp(argv[3+idx], "yolov3")) {
                name = "yolov3";
                period.tv_sec = YOLOV3_SEC;
                period.tv_nsec = YOLOV3_NSEC;
                prior = YOLO_PRIOR;
        
            } else if(0==strcmp(argv[3+idx], "yolov4")) {
                name = "yolov4";
                period.tv_sec = YOLOV4_SEC;
                period.tv_nsec = YOLOV4_NSEC;
                prior = YOLO_PRIOR;
        
            } else if(0==strcmp(argv[3+idx], "yolov2-tiny")) {
                name = "yolov2-tiny";
                period.tv_sec = YOLOV2T_SEC;
                period.tv_nsec = YOLOV2T_NSEC;
                prior = YOLO_PRIOR;
        
            } else if(0==strcmp(argv[3+idx], "yolov3-tiny")) {
                name = "yolov3-tiny";
                period.tv_sec = YOLOV3T_SEC;
                period.tv_nsec = YOLOV3T_NSEC;
                prior = YOLO_PRIOR;
        
            } else if(0==strcmp(argv[3+idx], "yolov4-tiny")) {
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
            dnn_buffer[idx].name = name;
            dnn_buffer[idx].prior = prior;
        
            printf("- This part create DNN: \n");
            printf("       Name: %s\n", dnn_buffer[idx].name);
            printf("     Period: %ld ms\n", dnn_buffer[idx].period.tv_nsec/1000000);
            printf("   Priority: %d\n", dnn_buffer[idx].prior);
            
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
            
            char *datacfg = argv[5];
            char *cfg = argv[7];
            char *weights = (argc > 8) ? argv[8] : 0;
            if (weights)
                if (strlen(weights) > 0)
                    if (weights[strlen(weights) - 1] == 0x0d) weights[strlen(weights) - 1] = 0;
            char *filename = (argc > 11) ? argv[11] : 0;
        
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
        
            int err = pthread_create(&demo_thread[idx], NULL, demo_detector_thread, (void *) args);
            if (err < 0) {
                perror("Detector thread create error : ");
                exit(0);
            }
        
            while (!dnn_buffer[idx].on) {};
            free(args);
        
            printf("---------------------------- Creat Complete Detector\n");


        } else { // Classifier: alexnet, darknet19
            int idx = i;
            printf("-------------------------------------------\n");
            printf("Classification Create Part %d\n", idx);
        
            char *name=NULL;
            int prior = 0;
            struct timespec period;
        
            if(0==strcmp(argv[3+idx], "alexnet")) {
                name = "alexnet";
                period.tv_sec = ALEXNET_SEC;
                period.tv_nsec = ALEXNET_NSEC;
                prior = ALEXNET_PRIOR;
        
            } else if(0==strcmp(argv[3+idx], "darknet19")) {
                name = "darknet19";
                period.tv_sec = DARKNET19_SEC;
                period.tv_nsec = DARKNET19_NSEC;
                prior = DARKNET19_PRIOR;
        
            } else if(0==strcmp(argv[3+idx], "densenet201")) {
                name = "densenet201";
                period.tv_sec = DENSENET201_SEC;
                period.tv_nsec = DENSENET201_NSEC;
                prior = DENSENET201_PRIOR;
        
            } else {
                perror("No match Network ");
                exit(0);
            }
        
            dnn_buffer[idx].on = 0;
            dnn_buffer[idx].period = period;
            dnn_buffer[idx].name = name;
            dnn_buffer[idx].prior = prior;
        
            printf("- This part create DNN: \n");
            printf("       Name: %s\n", dnn_buffer[idx].name);
            printf("     Period: %ld ms\n", dnn_buffer[idx].period.tv_nsec/1000000);
            printf("   Priority: %d\n", dnn_buffer[idx].prior);
            
        
        
            int benchmark = find_arg(argc, argv, "-benchmark");
            int benchmark_layers = find_arg(argc, argv, "-benchmark_layers");
            if (benchmark_layers) benchmark = 1;
            int cam_index = find_int_arg(argc, argv, "-c", 0);
            char *data = argv[6];
            char *cfg = argv[7 + 2*idx];
            char *weights = argv[7 + 2*idx + 1];
            char *filename = (argc > 11) ? argv[11]: 0;
        
            DemoClassi *args = (DemoClassi *)malloc(sizeof(DemoClassi));
            
            args->idx = idx;
            args->datacfg = data;
            args->cfgfile = cfg;
            args->weightfile = weights;
            args->cam_index = cam_index;
            args->filename = filename;
            args->benchmark = benchmark;
            args->benchmark_layers = benchmark_layers;
        
            int err = pthread_create(&demo_thread[idx], NULL, demo_classification_thread, (void *) args);
            if (err < 0) {
                perror("Detector thread create error : ");
                exit(0);
            }
        
            while (!dnn_buffer[idx].on) {};
            free(args);
        
            printf("---------------------------- Creat Complete\n");
        } // if else end
        sleep(1);

    } // for end

    custom_thread_t fetch_thread = NULL;
    custom_thread_t detect_thread = NULL;
    custom_thread_t classification_thread = NULL;
    custom_thread_t display_thread = NULL;
    printf("thesis check before create fetch\n");
    if (custom_create_thread(&fetch_thread, 0, multi_fetch_in_thread, 0)) error("Thread creation failed", DARKNET_LOC);
    printf("thesis check before create detect\n");
    if (custom_create_thread(&detect_thread, 0, multi_detect_in_thread, 0)) error("Thread creation failed", DARKNET_LOC);
    
    printf("thesis check before create classification\n");
    if (custom_create_thread(&classification_thread, 0, multi_classification_in_thread, 0)) error("Thread creation failed", DARKNET_LOC);

    printf("thesis check before create display\n");
    if (custom_create_thread(&display_thread, 0, multi_display_in_thread, 0)) error("Thread creation failed", DARKNET_LOC);


    free(args);

    int start_flag = 1;
    printf("\n\n\n\n\n");

    while(!dnn_buffer[0].on) {};
// mutex main

    struct timespec period;
    period.tv_sec = QUANTUM_SEC;
    period.tv_nsec = QUANTUM_NSEC;

    double detector_timer = 0;
    double classifier_timer = 0;

    int meas_counter = 0;
    int meas = 0;
    double present_quantum_time = 0;
    int classifier_count = 0;
    int detector_count = 0;
    
    while(1) {
        
//        printf("\n\n\nThesis Check Start   Main Quantum Counter\n\n\n");
        if(dnn_buffer[0].on && dnn_buffer[1].on && webcam_is_ok) {

//            printf("\n\n\nThesis Check Running Main Quantum Counter\n\n\n");
//            printf("Thesis meas_counter: %d\n", meas_counter);
            double start_quantum = multi_get_wall_time();
            detector_timer += (present_quantum_time);
            classifier_timer += (present_quantum_time);

#ifdef PRIORITY_H
            if (classifier_timer >= CLASSIFIER_PERIOD) {
//                printf("\n\n\nClassifier Release, Time: %.4lf\n\n\n", multi_get_wall_time());

                dnn_buffer[CLA].count = classifier_count;
                cla_count = classifier_count;

                custom_atomic_store_int(&run_classification_in_thread, 1);
                dnn_buffer[CLA].release_period[classifier_count] = classifier_timer;
                dnn_buffer[CLA].release_time[classifier_count] = multi_get_wall_time();
//                printf("Classifier count: %d, release: %.4lf\n", classifier_count, dnn_buffer[CLA].release_time[classifier_count]);

                classifier_timer = 0;
                classifier_count += 1;
            }
#endif

            if (detector_timer >= DETECTOR_PERIOD) {
//                printf("\n\n\nDetector Release, Time: %.1lf\n\n\n", multi_get_wall_time());

                dnn_buffer[DET].count = detector_count;
                det_count = detector_count;

                custom_atomic_store_int(&run_detect_in_thread, 1);
                dnn_buffer[DET].release_period[detector_count] = detector_timer;
                dnn_buffer[DET].release_time[detector_count] = multi_get_wall_time();

                detector_count += 1;
                detector_timer = 0;   
            }
#ifndef PRIORITY_H
            if (classifier_timer >= CLASSIFIER_PERIOD) {
//                printf("\n\n\nClassifier Release, Time: %.4lf\n\n\n", multi_get_wall_time());

                dnn_buffer[CLA].count = classifier_count;
                cla_count = classifier_count;

                custom_atomic_store_int(&run_classification_in_thread, 1);
                dnn_buffer[CLA].release_period[classifier_count] = classifier_timer;
                dnn_buffer[CLA].release_time[classifier_count] = multi_get_wall_time();
//                printf("Classifier count: %d, release: %.4lf\n", classifier_count, dnn_buffer[CLA].release_time[classifier_count]);

                classifier_timer = 0;
                classifier_count += 1;
            }
#endif

#ifdef MEASUREMENT
//            meas = dnn_buffer[0].count - MEAS_THRESHOLD;
//            meas = dnn_buffer[1].count;
            meas = classifier_count;
/*            
            if (meas >= 0) {
                meas_quantum_time_array[meas] = present_quantum_time;


            }
*/
            if (meas >= MEASUREMENT_ITERATION - 1) {
                system_exit_flag = 1;
                int exist = 0;
                FILE *fp;
                char file_path_d[100] = "";
                char file_path_c[100] = "";

                strcat(file_path_d, MEASUREMENT_PATH);
                strcat(file_path_d, MEASUREMENTD_FILE);

                fp = fopen(file_path_d, "w+");

                if (fp == NULL) {
                    int result;

                    result = mkdir(MEASUREMENT_PATH, 0766);

                    if(result == 0) {
                        exist = 1;
                        fp = fopen(file_path_d, "w+");
                    }
                }

                fprintf(fp, "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", "D_Period", "D_Release", "D_FrameTimestamp", "D_OnRunning",
                "D_BeforePrediction", "D_AfterPrediction", "D_DetecTime", "D_Complete", "D_DisplayStart", "D_DisplayEnd");

                for (int i = 0; i < dnn_buffer[DET].count; ++i) {
                    fprintf(fp, "%.1lf,%.1lf,%.1lf,%.1lf,%.1lf,%.1lf,%.1lf,%.1lf,%.1lf,%.1lf\n", 
                    dnn_buffer[DET].release_period[i], dnn_buffer[DET].release_time[i], dnn_buffer[DET].frame_timestamp[i],
                    dnn_buffer[DET].onrunning_time[i],
                    dnn_buffer[DET].before_prediction[i], dnn_buffer[DET].after_prediction[i],
                    dnn_buffer[DET].after_prediction[i] - dnn_buffer[DET].before_prediction[i],
                    dnn_buffer[DET].complete_time[i],
                    dnn_buffer[DET].display_start[i], dnn_buffer[DET].display_end[i]);
                }

                fclose(fp);

                
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

                fprintf(fp, "%s,%s,%s,%s,%s,%s,%s,%s\n", "C_Period", "C_Release", "C_FrameTimestamp", "C_OnRunning",
                "C_BeforePrediction", "C_AfterPrediction", "C_DetecTime", "C_Complete");

                for (int i = 0; i < dnn_buffer[CLA].count; ++i) {
                    fprintf(fp, "%.1lf,%.1lf,%.1lf,%.1lf,%.1lf,%.1lf,%.1lf,%.1lf\n", 
                    dnn_buffer[CLA].release_period[i], dnn_buffer[CLA].release_time[i], dnn_buffer[CLA].frame_timestamp[i],
                    dnn_buffer[CLA].onrunning_time[i],
                    dnn_buffer[CLA].before_prediction[i], dnn_buffer[CLA].after_prediction[i], 
                    dnn_buffer[CLA].after_prediction[i] - dnn_buffer[CLA].before_prediction[i],
                    dnn_buffer[CLA].complete_time[i]);
                }

                fclose(fp);

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

