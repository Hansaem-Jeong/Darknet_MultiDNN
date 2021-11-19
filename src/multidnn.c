#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "multidnn.h"
#include <time.h>

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

    printf("This part create DNN: \n");
    printf("       Name: %s\n", dnn_buffer[idx].name);
    printf("     Period: %ld ms\n", dnn_buffer[idx].period.tv_nsec/1000000);
    printf("   Priority: %d\n", dnn_buffer[idx].prior);
    

    printf("---------------------------- Creat Complete\n");
//    printf("idx: %d, argc: %d, argv: %s\n",
//            idx, argc, argv[1]);

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

    printf("This part create DNN: \n");
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


void run_multidnn(int argc, char **argv)
{
    int numberof_dnn = atoi(argv[2]);
    pthread_t p_thread[3];
    Args *args = (Args *)malloc(sizeof(Args));
    
    args->argc = argc;
    args->argv = argv;

    printf("Run Multi-DNN\n");

    for (int i = 0 ; i < numberof_dnn ; ++i) {
        args->idx = i;
        if (i == 0) { // Detector: yolo
            int err = pthread_create(&p_thread[i], NULL, set_detection_thread, (void *) args);
            if (err < 0) {
                perror("Detector thread create error : ");
                exit(0);
            }

        } else { // Classifier: alexnet, darknet19
            int err = pthread_create(&p_thread[i], NULL, set_classification_thread, (void *) args);
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

        if(start_flag) {
            start_flag = 0;
            for(int i=0;i<numberof_dnn;++i) {
                dnn_buffer[i].on = 1;
            }
        }
        sleep(1000);

    }
}

