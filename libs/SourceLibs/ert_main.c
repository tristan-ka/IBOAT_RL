/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * File: ert_main.c
 *
 * Code generated for Simulink model 'Model_2_EXPORT_Discrete'.
 *
 * Model version                  : 1.186
 * Simulink Coder version         : 8.12 (R2017a) 16-Feb-2017
 * C/C++ source code generated on : Wed Feb 07 14:28:56 2018
 *
 * Target selection: ert.tlc
 * Embedded hardware selection: ARM Compatible->ARM Cortex
 * Code generation objective: Execution efficiency
 * Validation result: Not run
 */

#include <stddef.h>
#include <stdio.h>                     /* This ert_main.c example uses printf/fflush */
#include "Model_2_EXPORT_Discrete.h"   /* Model's header file */
#include "rtwtypes.h"

static RT_MODEL_Model_2_EXPORT_Discr_T Model_2_EXPORT_Discrete_M_;
static RT_MODEL_Model_2_EXPORT_Discr_T *const Model_2_EXPORT_Discrete_M =
  &Model_2_EXPORT_Discrete_M_;         /* Real-time model */
static DW_Model_2_EXPORT_Discrete_T Model_2_EXPORT_Discrete_DW;/* Observable states */

/* '<Root>/sailpos' */
static real_T Model_2_EXPORT_Discrete_U_sailpos;

/* '<Root>/rudderpos' */
static real_T Model_2_EXPORT_Discrete_U_rudderpos;

/* '<Root>/true wind speed' */
static real_T Model_2_EXPORT_Discrete_U_truewindspeed;

/* '<Root>/true wind heading' */
static real_T Model_2_EXPORT_Discrete_U_truewindheading;

/* '<Root>/true water speed' */
static real_T Model_2_EXPORT_Discrete_U_truewaterspeed;

/* '<Root>/true water heading' */
static real_T Model_2_EXPORT_Discrete_U_truewaterheading;

/* '<Root>/hdg0' */
static real_T Model_2_EXPORT_Discrete_U_hdg0;

/* '<Root>/speed0' */
static real_T Model_2_EXPORT_Discrete_U_speed0;

/* '<Root>/Wind incidence' */
static real_T Model_2_EXPORT_Discrete_Y_Windincidence;

/* '<Root>/Speed Over Ground' */
static real_T Model_2_EXPORT_Discrete_Y_SpeedOverGround;

/*
 * Associating rt_OneStep with a real-time clock or interrupt service routine
 * is what makes the generated code "real-time".  The function rt_OneStep is
 * always associated with the base rate of the model.  Subrates are managed
 * by the base rate from inside the generated code.  Enabling/disabling
 * interrupts and floating point context switches are target specific.  This
 * example code indicates where these should take place relative to executing
 * the generated code step function.  Overrun behavior should be tailored to
 * your application needs.  This example simply sets an error status in the
 * real-time model and returns from rt_OneStep.
 */
void rt_OneStep(RT_MODEL_Model_2_EXPORT_Discr_T *const Model_2_EXPORT_Discrete_M);
void rt_OneStep(RT_MODEL_Model_2_EXPORT_Discr_T *const Model_2_EXPORT_Discrete_M)
{
  static boolean_T OverrunFlag = false;

  /* Disable interrupts here */

  /* Check for overrun */
  if (OverrunFlag) {
    return;
  }

  OverrunFlag = true;

  /* Save FPU context here (if necessary) */
  /* Re-enable timer or interrupt here */
  /* Set model inputs here */

  /* Step the model */
  Model_2_EXPORT_Discrete_step(Model_2_EXPORT_Discrete_M,
    Model_2_EXPORT_Discrete_U_sailpos, Model_2_EXPORT_Discrete_U_rudderpos,
    Model_2_EXPORT_Discrete_U_truewindspeed,
    Model_2_EXPORT_Discrete_U_truewindheading,
    Model_2_EXPORT_Discrete_U_truewaterspeed,
    Model_2_EXPORT_Discrete_U_truewaterheading, Model_2_EXPORT_Discrete_U_hdg0,
    Model_2_EXPORT_Discrete_U_speed0, &Model_2_EXPORT_Discrete_Y_Windincidence,
    &Model_2_EXPORT_Discrete_Y_SpeedOverGround);

  /* Get model outputs here */

  /* Indicate task complete */
  OverrunFlag = false;

  /* Disable interrupts here */
  /* Restore FPU context here (if necessary) */
  /* Enable interrupts here */
}

/*
 * The example "main" function illustrates what is required by your
 * application code to initialize, execute, and terminate the generated code.
 * Attaching rt_OneStep to a real-time clock is target specific.  This example
 * illustrates how you do this relative to initializing the model.
 */
int_T main(int_T argc, const char *argv[])
{
  /* Unused arguments */
  (void)(argc);
  (void)(argv);

  /* Pack model data into RTM */
  Model_2_EXPORT_Discrete_M->dwork = &Model_2_EXPORT_Discrete_DW;

  /* Initialize model */
  Model_2_EXPORT_Discrete_initialize(Model_2_EXPORT_Discrete_M,
    &Model_2_EXPORT_Discrete_U_sailpos, &Model_2_EXPORT_Discrete_U_rudderpos,
    &Model_2_EXPORT_Discrete_U_truewindspeed,
    &Model_2_EXPORT_Discrete_U_truewindheading,
    &Model_2_EXPORT_Discrete_U_truewaterspeed,
    &Model_2_EXPORT_Discrete_U_truewaterheading, &Model_2_EXPORT_Discrete_U_hdg0,
    &Model_2_EXPORT_Discrete_U_speed0, &Model_2_EXPORT_Discrete_Y_Windincidence,
    &Model_2_EXPORT_Discrete_Y_SpeedOverGround);

  /* Attach rt_OneStep to a timer or interrupt service routine with
   * period 0.01 seconds (the model's base sample time) here.  The
   * call syntax for rt_OneStep is
   *
   *  rt_OneStep(Model_2_EXPORT_Discrete_M);
   */
  printf("Warning: The simulation will run forever. "
         "Generated ERT main won't simulate model step behavior. "
         "To change this behavior select the 'MAT-file logging' option.\n");
  fflush((NULL));
  while (1) {
    /*  Perform other application tasks here */
  }

  /* The option 'Remove error status field in real-time model data structure'
   * is selected, therefore the following code does not need to execute.
   */
#if 0

  /* Disable rt_OneStep() here */

  /* Terminate model */
  Model_2_EXPORT_Discrete_terminate(Model_2_EXPORT_Discrete_M);

#endif

  return 0;
}

/*
 * File trailer for generated code.
 *
 * [EOF]
 */
