/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * File: Model_2_EXPORT_Discrete.h
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

#ifndef RTW_HEADER_Model_2_EXPORT_Discrete_h_
#define RTW_HEADER_Model_2_EXPORT_Discrete_h_
#include <float.h>
#include <math.h>
#include <string.h>
#ifndef Model_2_EXPORT_Discrete_COMMON_INCLUDES_
# define Model_2_EXPORT_Discrete_COMMON_INCLUDES_
#include "rtwtypes.h"
#endif                                 /* Model_2_EXPORT_Discrete_COMMON_INCLUDES_ */

#include "Model_2_EXPORT_Discrete_types.h"
#include "rt_look.h"
#include "rt_look1d.h"
#include "rt_defines.h"

/* Macros for accessing real-time model data structure */

/* Block states (auto storage) for system '<Root>' */
typedef struct {
  real_T incidencemeasure_DSTATE;      /* '<S8>/incidence measure' */
  real_T speedmeasure_DSTATE;          /* '<S3>/speed measure' */
  real_T FixPtUnitDelay1_DSTATE;       /* '<S11>/FixPt Unit Delay1' */
  real_T DiscreteTimeIntegrator1_DSTATE;/* '<S4>/Discrete-Time Integrator1' */
  real_T DiscreteTimeIntegrator_DSTATE;/* '<S27>/Discrete-Time Integrator' */
  real_T DiscreteTimeIntegrator_DSTATE_p;/* '<S4>/Discrete-Time Integrator' */
  real_T DiscreteFilter_states;        /* '<S4>/Discrete Filter' */
  real_T Discretemotormodel1_states;   /* '<S7>/Discrete motor model1' */
  real_T Discretemotormodel_states;    /* '<S6>/Discrete motor model' */
  real_T PrevY;                        /* '<S7>/Speed Limiter' */
  real_T PrevY_d;                      /* '<S6>/Speed Limiter' */
  uint8_T FixPtUnitDelay2_DSTATE;      /* '<S11>/FixPt Unit Delay2' */
  uint8_T DiscreteTimeIntegrator_IC_LOADI;/* '<S27>/Discrete-Time Integrator' */
  uint8_T DiscreteTimeIntegrator_IC_LOA_m;/* '<S4>/Discrete-Time Integrator' */
  boolean_T G3_PreviousInput;          /* '<S39>/G3' */
} DW_Model_2_EXPORT_Discrete_T;

/* Constant parameters (auto storage) */
typedef struct {
  /* Pooled Parameter (Mixed Expressions)
   * Referenced by:
   *   '<S28>/Angle2drag'
   *   '<S36>/décroché'
   */
  real_T pooled2[16];

  /* Expression: [0
     0.1075
     0.2196
     0.3391
     0.3905
     0.3900
     0.4267
     0.4251
     0.4248
     0.4398
     0.4567
     0.4783
     0.4959
     0.5121
     0.5003
     0.5181]
   * Referenced by: '<S36>/décroché'
   */
  real_T dcroch_YData[16];

  /* Expression: [0;2;4;6;8;10;12;14]*pi/180
   * Referenced by: '<S36>/accroché'
   */
  real_T accroch_XData[8];

  /* Expression: [0;0.112533129751682;0.223609894514084;0.332737416028976;0.453826665878296;0.563470125198364;0.661327242851257;0.728185296058655]
   * Referenced by: '<S36>/accroché'
   */
  real_T accroch_YData[8];

  /* Expression: [    0.0260
     0.0390
     0.0505
     0.0695
     0.0838
     0.1076
     0.1301
     0.1738
     0.2319
     0.2508
     0.2845
     0.3078
     0.3351
     0.3633
     0.3886
     0.4171]
   * Referenced by: '<S28>/Angle2drag'
   */
  real_T Angle2drag_YData[16];

  /* Expression: [-3.1416     -1.5677     -1.4755     -1.4006      -1.318     -1.1993     -1.1291     -1.0642    -0.97814     -0.8796    -0.84168    -0.77486    -0.72128    -0.64969    -0.53227    -0.44686    -0.37346    -0.27824    -0.17616    -0.12196           0     0.12196     0.17616     0.27824     0.37346     0.44686     0.53227     0.64969     0.72128     0.77486     0.84168      0.8796     0.97814      1.0642      1.1291      1.1993       1.318      1.4006      1.4755      1.5677      3.1416]
   * Referenced by: '<S30>/Angle2lift'
   */
  real_T Angle2lift_XData[41];

  /* Expression: [0  -0.0037453     -0.1161    -0.20599    -0.30337    -0.43446    -0.49438    -0.53558      -0.603    -0.68539    -0.70412    -0.72285    -0.73783    -0.79401    -0.88764    -0.90637    -0.86891    -0.79775    -0.70787    -0.54682           0     0.54682     0.70787     0.79775     0.86891     0.90637     0.88764     0.79401     0.73783     0.72285     0.70412     0.68539       0.603     0.53558     0.49438     0.43446     0.30337     0.20599      0.1161   0.0037453           0]
   * Referenced by: '<S30>/Angle2lift'
   */
  real_T Angle2lift_YData[41];

  /* Expression: [-3.1416     -1.5677     -1.4755     -1.4006      -1.318     -1.1993     -1.1291     -1.0642    -0.97814     -0.8796    -0.84168    -0.77486    -0.72128    -0.64969    -0.53227    -0.44686    -0.37346    -0.27824    -0.17616    -0.12196             0.12196     0.17616     0.27824     0.37346     0.44686     0.53227     0.64969     0.72128     0.77486     0.84168      0.8796     0.97814      1.0642      1.1291      1.1993       1.318      1.4006      1.4755      1.5677      3.1416]
   * Referenced by: '<S30>/Angle2drag1'
   */
  real_T Angle2drag1_XData[40];

  /* Expression: [1.2198      1.2198      1.2145      1.1984      1.1743      1.1153      1.0456     0.96515     0.89544     0.82842      0.7882     0.70777     0.64879     0.60322     0.52279     0.43432     0.34048     0.22788     0.12601    0.067024     0.067024     0.12601     0.22788     0.34048     0.43432     0.52279     0.60322     0.64879     0.70777      0.7882     0.82842     0.89544     0.96515      1.0456      1.1153      1.1743      1.1984      1.2145      1.2198      1.2198]
   * Referenced by: '<S30>/Angle2drag1'
   */
  real_T Angle2drag1_YData[40];
} ConstP_Model_2_EXPORT_Discret_T;

/* Real-time Model Data Structure */
struct tag_RTM_Model_2_EXPORT_Discre_T {
  DW_Model_2_EXPORT_Discrete_T *dwork;
};

/* Constant parameters (auto storage) */
extern const ConstP_Model_2_EXPORT_Discret_T Model_2_EXPORT_Discrete_ConstP;

/* Model entry point functions */
extern void Model_2_EXPORT_Discrete_initialize(RT_MODEL_Model_2_EXPORT_Discr_T *
  const Model_2_EXPORT_Discrete_M, real_T *Model_2_EXPORT_Discrete_U_sailpos,
  real_T *Model_2_EXPORT_Discrete_U_rudderpos, real_T
  *Model_2_EXPORT_Discrete_U_truewindspeed, real_T
  *Model_2_EXPORT_Discrete_U_truewindheading, real_T
  *Model_2_EXPORT_Discrete_U_truewaterspeed, real_T
  *Model_2_EXPORT_Discrete_U_truewaterheading, real_T
  *Model_2_EXPORT_Discrete_U_hdg0, real_T *Model_2_EXPORT_Discrete_U_speed0,
  real_T *Model_2_EXPORT_Discrete_Y_Windincidence, real_T
  *Model_2_EXPORT_Discrete_Y_SpeedOverGround);
extern void Model_2_EXPORT_Discrete_step(RT_MODEL_Model_2_EXPORT_Discr_T *const
  Model_2_EXPORT_Discrete_M, real_T Model_2_EXPORT_Discrete_U_sailpos, real_T
  Model_2_EXPORT_Discrete_U_rudderpos, real_T
  Model_2_EXPORT_Discrete_U_truewindspeed, real_T
  Model_2_EXPORT_Discrete_U_truewindheading, real_T
  Model_2_EXPORT_Discrete_U_truewaterspeed, real_T
  Model_2_EXPORT_Discrete_U_truewaterheading, real_T
  Model_2_EXPORT_Discrete_U_hdg0, real_T Model_2_EXPORT_Discrete_U_speed0,
  real_T *Model_2_EXPORT_Discrete_Y_Windincidence, real_T
  *Model_2_EXPORT_Discrete_Y_SpeedOverGround);
extern void Model_2_EXPORT_Discrete_terminate(RT_MODEL_Model_2_EXPORT_Discr_T *
  const Model_2_EXPORT_Discrete_M);

/*-
 * These blocks were eliminated from the model due to optimizations:
 *
 * Block '<S11>/FixPt Data Type Duplicate1' : Unused code path elimination
 * Block '<S22>/x->theta' : Unused code path elimination
 * Block '<S3>/Quantizer2' : Unused code path elimination
 * Block '<S23>/Constant1' : Unused code path elimination
 * Block '<S23>/Math Function' : Unused code path elimination
 * Block '<S3>/course measure' : Unused code path elimination
 * Block '<S28>/Scope' : Unused code path elimination
 * Block '<S42>/x->r' : Unused code path elimination
 * Block '<S8>/Quantizer2' : Unused code path elimination
 * Block '<S8>/speed measure' : Unused code path elimination
 * Block '<Root>/Scope1' : Unused code path elimination
 */

/*-
 * The generated code includes comments that allow you to trace directly
 * back to the appropriate location in the model.  The basic format
 * is <system>/block_name, where system is the system number (uniquely
 * assigned by Simulink) and block_name is the name of the block.
 *
 * Use the MATLAB hilite_system command to trace the generated code back
 * to the model.  For example,
 *
 * hilite_system('<S3>')    - opens system 3
 * hilite_system('<S3>/Kp') - opens and selects block Kp which resides in S3
 *
 * Here is the system hierarchy for this model
 *
 * '<Root>' : 'Model_2_EXPORT_Discrete'
 * '<S1>'   : 'Model_2_EXPORT_Discrete/Compass'
 * '<S2>'   : 'Model_2_EXPORT_Discrete/External conditions'
 * '<S3>'   : 'Model_2_EXPORT_Discrete/GPS'
 * '<S4>'   : 'Model_2_EXPORT_Discrete/Physics'
 * '<S5>'   : 'Model_2_EXPORT_Discrete/Radians to Degrees1'
 * '<S6>'   : 'Model_2_EXPORT_Discrete/Rudder Actuator'
 * '<S7>'   : 'Model_2_EXPORT_Discrete/Sail Actuator'
 * '<S8>'   : 'Model_2_EXPORT_Discrete/Weather Station'
 * '<S9>'   : 'Model_2_EXPORT_Discrete/Compass/Degrees to Radians2'
 * '<S10>'  : 'Model_2_EXPORT_Discrete/Compass/Subsystem2'
 * '<S11>'  : 'Model_2_EXPORT_Discrete/Compass/heading measure'
 * '<S12>'  : 'Model_2_EXPORT_Discrete/External conditions/Water over ground'
 * '<S13>'  : 'Model_2_EXPORT_Discrete/External conditions/Wind over ground'
 * '<S14>'  : 'Model_2_EXPORT_Discrete/External conditions/Water over ground/Conversion'
 * '<S15>'  : 'Model_2_EXPORT_Discrete/External conditions/Water over ground/Degrees to Radians1'
 * '<S16>'  : 'Model_2_EXPORT_Discrete/External conditions/Water over ground/Polar to Cartesian'
 * '<S17>'  : 'Model_2_EXPORT_Discrete/External conditions/Wind over ground/Band-Limited White Noise'
 * '<S18>'  : 'Model_2_EXPORT_Discrete/External conditions/Wind over ground/Band-Limited White Noise1'
 * '<S19>'  : 'Model_2_EXPORT_Discrete/External conditions/Wind over ground/Conversion'
 * '<S20>'  : 'Model_2_EXPORT_Discrete/External conditions/Wind over ground/Degrees to Radians2'
 * '<S21>'  : 'Model_2_EXPORT_Discrete/External conditions/Wind over ground/Polar to Cartesian'
 * '<S22>'  : 'Model_2_EXPORT_Discrete/GPS/Cartesian to Polar'
 * '<S23>'  : 'Model_2_EXPORT_Discrete/GPS/Subsystem'
 * '<S24>'  : 'Model_2_EXPORT_Discrete/Physics/BoatRelativeWater'
 * '<S25>'  : 'Model_2_EXPORT_Discrete/Physics/BoatRelativeWind'
 * '<S26>'  : 'Model_2_EXPORT_Discrete/Physics/Conversion'
 * '<S27>'  : 'Model_2_EXPORT_Discrete/Physics/RudderAction'
 * '<S28>'  : 'Model_2_EXPORT_Discrete/Physics/Subsystem1'
 * '<S29>'  : 'Model_2_EXPORT_Discrete/Physics/Subsystem2'
 * '<S30>'  : 'Model_2_EXPORT_Discrete/Physics/Subsystem3'
 * '<S31>'  : 'Model_2_EXPORT_Discrete/Physics/Subsystem4'
 * '<S32>'  : 'Model_2_EXPORT_Discrete/Physics/Subsystem5'
 * '<S33>'  : 'Model_2_EXPORT_Discrete/Physics/Subsystem6'
 * '<S34>'  : 'Model_2_EXPORT_Discrete/Physics/VVXY->VVxy1'
 * '<S35>'  : 'Model_2_EXPORT_Discrete/Physics/RudderAction/Degrees to Radians2'
 * '<S36>'  : 'Model_2_EXPORT_Discrete/Physics/Subsystem1/Hysteresis  '
 * '<S37>'  : 'Model_2_EXPORT_Discrete/Physics/Subsystem1/Hysteresis  /Degrees to Radians1'
 * '<S38>'  : 'Model_2_EXPORT_Discrete/Physics/Subsystem1/Hysteresis  /Degrees to Radians2'
 * '<S39>'  : 'Model_2_EXPORT_Discrete/Physics/Subsystem1/Hysteresis  /logique'
 * '<S40>'  : 'Model_2_EXPORT_Discrete/Rudder Actuator/Degrees to Radians1'
 * '<S41>'  : 'Model_2_EXPORT_Discrete/Sail Actuator/Degrees to Radians1'
 * '<S42>'  : 'Model_2_EXPORT_Discrete/Weather Station/Cartesian to Polar'
 * '<S43>'  : 'Model_2_EXPORT_Discrete/Weather Station/Subsystem1'
 */
#endif                                 /* RTW_HEADER_Model_2_EXPORT_Discrete_h_ */

/*
 * File trailer for generated code.
 *
 * [EOF]
 */
