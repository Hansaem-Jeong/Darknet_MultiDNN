#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "multidnn.h"
#include <time.h>

#include "network.h"
#include "utils.h"
#include "parser.h"
#include "option_list.h"
#include "blas.h"
#include "assert.h"
#include "dark_cuda.h"

#define ALEXNET_SEC 0
#define ALEXNET_NSEC 100000000 // 100ms
#define ALEXNET_PRIOR 5

#define DARKNET19_SEC 0
#define DARKNET19_NSEC 100000000 // 100ms
#define DARKNET19_PRIOR 6

#define DENSENET201_SEC 0
#define DENSENET201_NSEC 100000000 // 100ms
#define DENSENET201_PRIOR 7


#define YOLOV2_SEC 0
#define YOLOV2_NSEC 150000000 // 150ms
#define YOLOV3_SEC 0
#define YOLOV3_NSEC 150000000 // 150ms
#define YOLOV4_SEC 0
#define YOLOV4_NSEC 150000000 // 150ms
#define YOLO_PRIOR 10

volatile MultiDNN dnn_buffer[3];
pthread_t demo_thread[2];

// Classifier
void *set_classification_thread(void *arg)
{
    Args *argm = (Args *)arg;
    int idx = argm->idx;
    int argc = argm->argc;
    char **argv = argm->argv;

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
    printf("check data: %s\n", args->datacfg);
    args->cfgfile = cfg;
    args->weightfile = weights;
    args->cam_index = cam_index;
    args->filename = filename;
    args->benchmark = benchmark;
    args->benchmark_layers = benchmark_layers;

    int err = pthread_create(&demo_thread[idx-1], NULL, demo_classification_thread, (void *) args);
    if (err < 0) {
        perror("Detector thread create error : ");
        exit(0);
    }

//    free(args);

    printf("---------------------------- Creat Complete\n");
    while(1) {

        if(dnn_buffer[idx].on) {
            printf("idx: %d, dnn name: %s\n",
                    idx, dnn_buffer[idx].name);
            nanosleep(&dnn_buffer[idx].period, NULL);
            dnn_buffer[idx].release = 1; // += 1?
        }

    }

}

// Detector
void *set_detection_thread(void *arg)
{
    printf("-------------------------------------------\n");
    printf("Detector Create Part\n");
    Args *argm = (Args *)arg;
    int idx = argm->idx;
    int argc = argm->argc;
    char **argv = argm->argv;
 
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
    

    printf("---------------------------- Creat Complete\n");
    
    
    while(1) {

        if(dnn_buffer[idx].on) {
            printf("idx: %d, dnn name: %s\n",
                    idx, dnn_buffer[idx].name);
            nanosleep(&dnn_buffer[idx].period, NULL);
            dnn_buffer[idx].release = 1; // += 1?
        }

    }

}

void *demo_classification_thread(void *arg)
{
#ifdef OPENCV
    printf("Classifier Demo\n");

    DemoClassi *argm = (DemoClassi *)arg;
    int idx = argm->idx;
    char *datacfg = argm->datacfg;
    char *cfgfile = argm->cfgfile;
    char *weightfile = argm->weightfile;
    int cam_index = argm->cam_index;
    const char *filename = argm->filename;
    int benchmark = argm->benchmark;
    int benchmark_layers = argm->benchmark_layers;

    printf("       idx: %d\n", idx);
    printf("   datacfg: %s\n", datacfg);
    printf("   cfgfile: %s\n", cfgfile);
    printf("weightfile: %s\n", weightfile);
    printf(" cam_index: %d\n", cam_index);
    printf("  filename: %s\n", filename);

    network net = parse_network_cfg_custom(cfgfile, 1, 0);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    net.benchmark_layers = benchmark_layers;
    set_batch_network(&net, 1);
    list *options = read_data_cfg(datacfg);

    fuse_conv_batchnorm(net);
    calculate_binary_weights(net);

    srand(2222222);
    cap_cv * cap;

    if(filename){
        cap = get_capture_video_stream(filename);
    }else{
        cap = get_capture_webcam(cam_index);
    }

    int classes = option_find_int(options, "classes", 2);
    int top = option_find_int(options, "top", 1);
    if (top > classes) top = classes;

    char *name_list = option_find_str(options, "names", 0);
    char **names = get_labels(name_list);

    int* indexes = (int*)xcalloc(top, sizeof(int));

    if(!cap) error("Couldn't connect to webcam.", DARKNET_LOC);
    if (!benchmark) create_window_cv("Classifier", 0, 512, 512);
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

    while(1){
        if(dnn_buffer[idx].release) {
            dnn_buffer[idx].release = 0;
            printf("classification release\n");
        
            struct timeval tval_before, tval_after, tval_result;
            gettimeofday(&tval_before, NULL);
    
            //image in = get_image_from_stream(cap);
            image in_s, in;
            if (!benchmark) {
                in = get_image_from_stream_cpp(cap);
                in_s = resize_image(in, net.w, net.h);
                show_image(in, "Classifier");
            }
            else {
                static image tmp;
                if (!tmp.data) tmp = make_image(net.w, net.h, 3);
                in_s = tmp;
            }
    
            old_time = time;
            time = get_time_point();
            cycle_time = (time - old_time)/1000;
            float *predictions = network_predict(net, in_s.data);
            double frame_time_ms = (get_time_point() - time)/1000;
    // thesis
    //        printf("     cycle time : %lf\n", cycle_time);
    //        printf(" execution time : %lf\n", frame_time_ms);
    
            frame_counter++;
    // thesis
            thesis_idx += 1;      
    
            if(net.hierarchy) hierarchy_predictions(predictions, net.outputs, net.hierarchy, 1);
            top_predictions(net, top, indexes);
    
    #ifndef _WIN32
            printf("\033[2J");
            printf("\033[1;1H");
    #endif
    
    
            if (!benchmark) {
                printf("\ridx: %d, FPS: %.2f  (use -benchmark command line flag for correct measurement)\n", thesis_idx, fps);
                for (i = 0; i < top; ++i) {
                    int index = indexes[i];
                    printf("%.1f%%: %s\n", predictions[index] * 100, names[index]);
                }
                printf("\n");
    
                free_image(in_s);
                free_image(in);
    
                int c = wait_key_cv(10);// cvWaitKey(10);
                if (c == 27 || c == 1048603) break;
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
    
            if (spent_time >= 3.0f) {
                //printf(" spent_time = %f \n", spent_time);
                avg_fps = frame_counter / spent_time;
                frame_counter = 0;
                start_time = get_time_point();
            }
        }
    }

#endif
}


void run_multidnn(int argc, char **argv)
{
    int numberof_dnn = atoi(argv[2]);
    pthread_t set_thread[3];
    Args *args = (Args *)malloc(sizeof(Args));
    
    args->argc = argc;
    args->argv = argv;

    printf("Run Multi-DNN\n");

    for (int i = 0 ; i < numberof_dnn ; ++i) {
        args->idx = i;
        if (i == 0) { // Detector: yolo
            int err = pthread_create(&set_thread[i], NULL, set_detection_thread, (void *) args);
            if (err < 0) {
                perror("Detector thread create error : ");
                exit(0);
            }

        } else { // Classifier: alexnet, darknet19
            int err = pthread_create(&set_thread[i], NULL, set_classification_thread, (void *) args);
            if (err < 0) {
                perror("Classification thread create error : ");
                exit(0);
            }

        } // if else end
        sleep(1);

    } // for end



    free(args);

    int start_flag = 1;
    printf("\n\n\n\n\n");
    while(1) {
/*
        if(start_flag) {
            start_flag = 0;
            for(int i=0;i<numberof_dnn;++i) {
                dnn_buffer[i].on = 1;
            }
        }
*/
        sleep(1000);

    }
}

