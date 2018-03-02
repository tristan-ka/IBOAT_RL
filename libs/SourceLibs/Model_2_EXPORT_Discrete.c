/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * File: Model_2_EXPORT_Discrete.c
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

#include "Model_2_EXPORT_Discrete.h"
#include "Model_2_EXPORT_Discrete_private.h"

real_T rt_modd(real_T u0, real_T u1)
{
  real_T y;
  boolean_T yEq;
  real_T q;
  y = u0;
  if (u0 == 0.0) {
    y = 0.0;
  } else {
    if (u1 != 0.0) {
      y = fmod(u0, u1);
      yEq = (y == 0.0);
      if ((!yEq) && (u1 > floor(u1))) {
        q = fabs(u0 / u1);
        yEq = (fabs(q - floor(q + 0.5)) <= DBL_EPSILON * q);
      }

      if (yEq) {
        y = 0.0;
      } else {
        if ((u0 < 0.0) != (u1 < 0.0)) {
          y += u1;
        }
      }
    }
  }

  return y;
}

real_T rt_roundd(real_T u)
{
  real_T y;
  if (fabs(u) < 4.503599627370496E+15) {
    if (u >= 0.5) {
      y = floor(u + 0.5);
    } else if (u > -0.5) {
      y = 0.0;
    } else {
      y = ceil(u - 0.5);
    }
  } else {
    y = u;
  }

  return y;
}

real_T rt_hypotd(real_T u0, real_T u1)
{
  real_T y;
  real_T a;
  real_T b;
  a = fabs(u0);
  b = fabs(u1);
  if (a < b) {
    a /= b;
    y = sqrt(a * a + 1.0) * b;
  } else if (a > b) {
    b /= a;
    y = sqrt(b * b + 1.0) * a;
  } else {
    y = a * 1.4142135623730951;
  }

  return y;
}

/* Model step function */
void Model_2_EXPORT_Discrete_step(RT_MODEL_Model_2_EXPORT_Discr_T *const
  Model_2_EXPORT_Discrete_M, real_T Model_2_EXPORT_Discrete_U_sailpos, real_T
  Model_2_EXPORT_Discrete_U_rudderpos, real_T
  Model_2_EXPORT_Discrete_U_truewindspeed, real_T
  Model_2_EXPORT_Discrete_U_truewindheading, real_T
  Model_2_EXPORT_Discrete_U_truewaterspeed, real_T
  Model_2_EXPORT_Discrete_U_truewaterheading, real_T
  Model_2_EXPORT_Discrete_U_hdg0, real_T Model_2_EXPORT_Discrete_U_speed0,
  real_T *Model_2_EXPORT_Discrete_Y_Windincidence, real_T
  *Model_2_EXPORT_Discrete_Y_SpeedOverGround)
{
  DW_Model_2_EXPORT_Discrete_T *Model_2_EXPORT_Discrete_DW =
    ((DW_Model_2_EXPORT_Discrete_T *) Model_2_EXPORT_Discrete_M->dwork);
  real_T rtb_Vely;
  real_T rtb_VAy;
  real_T rtb_Velx;
  real_T rtb_Product1_d;
  real_T rtb_Product3_f;
  real_T rtb_TrigonometricFunction;
  real_T rtb_Sum2_e;
  real_T rtb_Sum_i;
  real_T rtb_Fl;
  real_T rtb_SpeedLimiter;
  real_T rtb_Product_p;
  real_T rtb_Product3_k;
  real_T TrigonometricFunction;
  real_T rtb_Flip_idx_1;
  real_T rtb_Flip_b_idx_1;
  real_T rtb_V2measure_idx_1;
  real_T rtb_Flip_idx_0;

  /* Outport: '<Root>/Wind incidence' incorporates:
   *  Constant: '<S43>/Constant1'
   *  Math: '<S43>/Math Function'
   *  Quantizer: '<S8>/Quantizer1'
   *  UnitDelay: '<S8>/incidence measure'
   */
  *Model_2_EXPORT_Discrete_Y_Windincidence = rt_roundd(rt_modd
    (Model_2_EXPORT_Discrete_DW->incidencemeasure_DSTATE, 6.2831853071795862) /
    0.0001) * 0.0001;

  /* Outport: '<Root>/Speed Over Ground' incorporates:
   *  Quantizer: '<S3>/Quantizer1'
   *  UnitDelay: '<S3>/speed measure'
   */
  *Model_2_EXPORT_Discrete_Y_SpeedOverGround = rt_roundd
    (Model_2_EXPORT_Discrete_DW->speedmeasure_DSTATE / 0.01) * 0.01;

  /* Gain: '<S19>/Gain2' incorporates:
   *  Gain: '<S13>/Gain1'
   *  Inport: '<Root>/true wind speed'
   */
  rtb_Vely = 0.51444444444444448 * -Model_2_EXPORT_Discrete_U_truewindspeed;

  /* Gain: '<S20>/Gain1' incorporates:
   *  Inport: '<Root>/true wind heading'
   */
  rtb_VAy = 0.017453292519943295 * Model_2_EXPORT_Discrete_U_truewindheading;

  /* Fcn: '<S21>/r->x' */
  rtb_Velx = rtb_Vely * cos(rtb_VAy);

  /* Fcn: '<S21>/theta->y' */
  rtb_Vely *= sin(rtb_VAy);

  /* DSPFlip: '<S13>/Flip' incorporates:
   *  SignalConversion: '<S13>/TmpSignal ConversionAtFlipInport1'
   */
  rtb_Flip_idx_0 = rtb_Vely;
  rtb_Flip_idx_1 = rtb_Velx;

  /* Gain: '<S14>/Gain2' incorporates:
   *  Gain: '<S12>/Gain'
   *  Inport: '<Root>/true water speed'
   */
  rtb_Velx = 0.51444444444444448 * -Model_2_EXPORT_Discrete_U_truewaterspeed;

  /* Gain: '<S15>/Gain1' incorporates:
   *  Inport: '<Root>/true water heading'
   */
  rtb_VAy = 0.017453292519943295 * Model_2_EXPORT_Discrete_U_truewaterheading;

  /* Fcn: '<S16>/r->x' */
  rtb_Vely = rtb_Velx * cos(rtb_VAy);

  /* Fcn: '<S16>/theta->y' */
  rtb_Velx *= sin(rtb_VAy);

  /* DSPFlip: '<S12>/Flip' incorporates:
   *  SignalConversion: '<S12>/TmpSignal ConversionAtFlipInport1'
   */
  rtb_Flip_b_idx_1 = rtb_Vely;

  /* DiscreteIntegrator: '<S27>/Discrete-Time Integrator' incorporates:
   *  Gain: '<S35>/Gain1'
   *  Inport: '<Root>/hdg0'
   */
  if (Model_2_EXPORT_Discrete_DW->DiscreteTimeIntegrator_IC_LOADI != 0) {
    Model_2_EXPORT_Discrete_DW->DiscreteTimeIntegrator_DSTATE =
      0.017453292519943295 * Model_2_EXPORT_Discrete_U_hdg0;
  }

  /* Trigonometry: '<S34>/Trigonometric Function1' incorporates:
   *  DiscreteIntegrator: '<S27>/Discrete-Time Integrator'
   */
  rtb_VAy = cos(Model_2_EXPORT_Discrete_DW->DiscreteTimeIntegrator_DSTATE);

  /* DiscreteIntegrator: '<S4>/Discrete-Time Integrator' incorporates:
   *  Gain: '<S26>/Gain2'
   *  Inport: '<Root>/speed0'
   */
  if (Model_2_EXPORT_Discrete_DW->DiscreteTimeIntegrator_IC_LOA_m != 0) {
    Model_2_EXPORT_Discrete_DW->DiscreteTimeIntegrator_DSTATE_p =
      0.51444444444444448 * Model_2_EXPORT_Discrete_U_speed0;
  }

  /* Trigonometry: '<S34>/Trigonometric Function' incorporates:
   *  DiscreteIntegrator: '<S27>/Discrete-Time Integrator'
   */
  rtb_Vely = sin(Model_2_EXPORT_Discrete_DW->DiscreteTimeIntegrator_DSTATE);

  /* SignalConversion: '<S3>/TmpSignal ConversionAtFlipInport1' incorporates:
   *  DiscreteIntegrator: '<S4>/Discrete-Time Integrator'
   *  DiscreteIntegrator: '<S4>/Discrete-Time Integrator1'
   *  Product: '<S34>/Product2'
   *  Product: '<S34>/Product3'
   *  Sum: '<S34>/Sum1'
   */
  rtb_V2measure_idx_1 =
    Model_2_EXPORT_Discrete_DW->DiscreteTimeIntegrator_DSTATE_p * rtb_VAy -
    Model_2_EXPORT_Discrete_DW->DiscreteTimeIntegrator1_DSTATE * rtb_Vely;

  /* DSPFlip: '<S3>/Flip' incorporates:
   *  DiscreteIntegrator: '<S4>/Discrete-Time Integrator'
   *  DiscreteIntegrator: '<S4>/Discrete-Time Integrator1'
   *  Product: '<S34>/Product'
   *  Product: '<S34>/Product1'
   *  Sum: '<S34>/Sum'
   */
  rtb_VAy = Model_2_EXPORT_Discrete_DW->DiscreteTimeIntegrator1_DSTATE * rtb_VAy
    + Model_2_EXPORT_Discrete_DW->DiscreteTimeIntegrator_DSTATE_p * rtb_Vely;

  /* Update for UnitDelay: '<S3>/speed measure' incorporates:
   *  DSPFlip: '<S3>/Flip'
   *  Fcn: '<S22>/x->r'
   */
  Model_2_EXPORT_Discrete_DW->speedmeasure_DSTATE = rt_hypotd
    (rtb_V2measure_idx_1, rtb_VAy);

  /* Trigonometry: '<S24>/Trigonometric Function1' incorporates:
   *  DiscreteIntegrator: '<S27>/Discrete-Time Integrator'
   */
  rtb_Vely = cos(Model_2_EXPORT_Discrete_DW->DiscreteTimeIntegrator_DSTATE);

  /* Product: '<S24>/Product' incorporates:
   *  SignalConversion: '<S12>/TmpSignal ConversionAtFlipInport1'
   */
  rtb_V2measure_idx_1 = rtb_Velx * rtb_Vely;

  /* Trigonometry: '<S24>/Trigonometric Function' incorporates:
   *  DiscreteIntegrator: '<S27>/Discrete-Time Integrator'
   */
  rtb_VAy = sin(Model_2_EXPORT_Discrete_DW->DiscreteTimeIntegrator_DSTATE);

  /* Product: '<S24>/Product1' */
  rtb_Product1_d = rtb_Flip_b_idx_1 * rtb_VAy;

  /* Product: '<S24>/Product2' incorporates:
   *  SignalConversion: '<S12>/TmpSignal ConversionAtFlipInport1'
   */
  rtb_Velx *= rtb_VAy;

  /* Product: '<S24>/Product3' */
  rtb_Flip_b_idx_1 *= rtb_Vely;

  /* Trigonometry: '<S25>/Trigonometric Function1' incorporates:
   *  DiscreteIntegrator: '<S27>/Discrete-Time Integrator'
   */
  rtb_Vely = cos(Model_2_EXPORT_Discrete_DW->DiscreteTimeIntegrator_DSTATE);

  /* Trigonometry: '<S25>/Trigonometric Function' incorporates:
   *  DiscreteIntegrator: '<S27>/Discrete-Time Integrator'
   */
  rtb_VAy = sin(Model_2_EXPORT_Discrete_DW->DiscreteTimeIntegrator_DSTATE);

  /* Product: '<S25>/Product3' */
  rtb_Product3_f = rtb_Flip_idx_1 * rtb_Vely;

  /* Sum: '<S4>/Sum' incorporates:
   *  DiscreteIntegrator: '<S4>/Discrete-Time Integrator1'
   *  Product: '<S25>/Product'
   *  Product: '<S25>/Product1'
   *  Sum: '<S25>/Sum'
   */
  rtb_Vely = (rtb_Flip_idx_0 * rtb_Vely - rtb_Flip_idx_1 * rtb_VAy) -
    Model_2_EXPORT_Discrete_DW->DiscreteTimeIntegrator1_DSTATE;

  /* Sum: '<S4>/Sum1' incorporates:
   *  DiscreteIntegrator: '<S4>/Discrete-Time Integrator'
   *  Product: '<S25>/Product2'
   *  Sum: '<S25>/Sum1'
   */
  rtb_VAy = (rtb_Flip_idx_0 * rtb_VAy + rtb_Product3_f) -
    Model_2_EXPORT_Discrete_DW->DiscreteTimeIntegrator_DSTATE_p;

  /* Product: '<S28>/Product' incorporates:
   *  Constant: '<S28>/Constant2'
   *  Constant: '<S28>/Constant3'
   *  DiscreteFilter: '<S4>/Discrete Filter'
   *  Math: '<S31>/Math Function1'
   *  Math: '<S31>/Math Function2'
   *  Sum: '<S31>/Sum2'
   *  Trigonometry: '<S28>/Trigonometric Function'
   *
   * About '<S31>/Math Function1':
   *  Operator: magnitude^2
   *
   * About '<S31>/Math Function2':
   *  Operator: magnitude^2
   */
  rtb_Flip_idx_0 = cos(0.00995 *
                       Model_2_EXPORT_Discrete_DW->DiscreteFilter_states) * 0.7 *
    0.65 * (rtb_Vely * rtb_Vely + rtb_VAy * rtb_VAy);

  /* Trigonometry: '<S31>/Trigonometric Function' incorporates:
   *  Gain: '<S31>/Gain'
   *  Gain: '<S31>/Gain1'
   */
  rtb_Product3_f = atan2(-rtb_Vely, -rtb_VAy);

  /* DiscreteFilter: '<S7>/Discrete motor model1' */
  rtb_Flip_idx_1 = 0.04877 *
    Model_2_EXPORT_Discrete_DW->Discretemotormodel1_states;

  /* RateLimiter: '<S7>/Speed Limiter' */
  TrigonometricFunction = rtb_Flip_idx_1 - Model_2_EXPORT_Discrete_DW->PrevY;
  if (TrigonometricFunction > 0.17170000000000002) {
    rtb_Flip_idx_1 = Model_2_EXPORT_Discrete_DW->PrevY + 0.17170000000000002;
  } else {
    if (TrigonometricFunction < -0.17170000000000002) {
      rtb_Flip_idx_1 = Model_2_EXPORT_Discrete_DW->PrevY + -0.17170000000000002;
    }
  }

  Model_2_EXPORT_Discrete_DW->PrevY = rtb_Flip_idx_1;

  /* End of RateLimiter: '<S7>/Speed Limiter' */

  /* Sum: '<S4>/Sum2' incorporates:
   *  Gain: '<S41>/Gain1'
   */
  rtb_Sum2_e = 0.017453292519943295 * rtb_Flip_idx_1 + rtb_Product3_f;

  /* Switch: '<S36>/Switch1' incorporates:
   *  Lookup: '<S36>/accroché'
   *  Lookup: '<S36>/décroché'
   *  Memory: '<S39>/G3'
   */
  if (Model_2_EXPORT_Discrete_DW->G3_PreviousInput) {
    rtb_Flip_idx_1 = rt_Lookup(Model_2_EXPORT_Discrete_ConstP.accroch_XData, 8,
      rtb_Sum2_e, Model_2_EXPORT_Discrete_ConstP.accroch_YData);
  } else {
    rtb_Flip_idx_1 = rt_Lookup(Model_2_EXPORT_Discrete_ConstP.pooled2, 16,
      rtb_Sum2_e, Model_2_EXPORT_Discrete_ConstP.dcroch_YData);
  }

  /* End of Switch: '<S36>/Switch1' */

  /* Product: '<S28>/Product1' */
  rtb_Flip_idx_1 *= rtb_Flip_idx_0;

  /* Trigonometry: '<S32>/Trigonometric Function1' */
  rtb_TrigonometricFunction = cos(rtb_Product3_f);

  /* Product: '<S28>/Product2' incorporates:
   *  Lookup: '<S28>/Angle2drag'
   */
  rtb_Flip_idx_0 *= rt_Lookup(Model_2_EXPORT_Discrete_ConstP.pooled2, 16,
    rtb_Sum2_e, Model_2_EXPORT_Discrete_ConstP.Angle2drag_YData);

  /* Trigonometry: '<S32>/Trigonometric Function' */
  rtb_Product3_f = sin(rtb_Product3_f);

  /* Sum: '<S32>/Sum' incorporates:
   *  Product: '<S32>/Product'
   *  Product: '<S32>/Product1'
   */
  rtb_Sum_i = (0.0 - rtb_Flip_idx_1 * rtb_TrigonometricFunction) -
    rtb_Flip_idx_0 * rtb_Product3_f;

  /* Sum: '<S4>/Sum5' incorporates:
   *  DiscreteIntegrator: '<S4>/Discrete-Time Integrator1'
   *  Sum: '<S24>/Sum'
   */
  rtb_V2measure_idx_1 =
    Model_2_EXPORT_Discrete_DW->DiscreteTimeIntegrator1_DSTATE -
    (rtb_V2measure_idx_1 - rtb_Product1_d);

  /* Sum: '<S4>/Sum6' incorporates:
   *  DiscreteIntegrator: '<S4>/Discrete-Time Integrator'
   *  Sum: '<S24>/Sum1'
   */
  rtb_Velx = Model_2_EXPORT_Discrete_DW->DiscreteTimeIntegrator_DSTATE_p -
    (rtb_Velx + rtb_Flip_b_idx_1);

  /* Sum: '<S29>/Sum2' incorporates:
   *  Math: '<S29>/Math Function1'
   *  Math: '<S29>/Math Function2'
   *
   * About '<S29>/Math Function1':
   *  Operator: magnitude^2
   *
   * About '<S29>/Math Function2':
   *  Operator: magnitude^2
   */
  rtb_Flip_b_idx_1 = rtb_V2measure_idx_1 * rtb_V2measure_idx_1 + rtb_Velx *
    rtb_Velx;

  /* Product: '<S30>/Product' incorporates:
   *  Constant: '<S30>/Constant2'
   *  Constant: '<S30>/Constant3'
   */
  rtb_Product1_d = 60.0 * rtb_Flip_b_idx_1;

  /* Trigonometry: '<S29>/Trigonometric Function' */
  rtb_V2measure_idx_1 = atan2(rtb_V2measure_idx_1, rtb_Velx);

  /* Product: '<S30>/Product1' incorporates:
   *  Lookup: '<S30>/Angle2lift'
   */
  rtb_Fl = rtb_Product1_d * rt_Lookup
    (Model_2_EXPORT_Discrete_ConstP.Angle2lift_XData, 41, rtb_V2measure_idx_1,
     Model_2_EXPORT_Discrete_ConstP.Angle2lift_YData);

  /* Trigonometry: '<S33>/Trigonometric Function1' */
  rtb_SpeedLimiter = cos(rtb_V2measure_idx_1);

  /* Product: '<S33>/Product' */
  rtb_Product_p = rtb_Fl * rtb_SpeedLimiter;

  /* Product: '<S30>/Product2' incorporates:
   *  Lookup: '<S30>/Angle2drag1'
   */
  rtb_Product1_d *= rt_Lookup(Model_2_EXPORT_Discrete_ConstP.Angle2drag1_XData,
    40, rtb_V2measure_idx_1, Model_2_EXPORT_Discrete_ConstP.Angle2drag1_YData);

  /* Trigonometry: '<S33>/Trigonometric Function' */
  rtb_V2measure_idx_1 = sin(rtb_V2measure_idx_1);

  /* Product: '<S33>/Product3' */
  rtb_Product3_k = rtb_Product1_d * rtb_SpeedLimiter;

  /* DiscreteFilter: '<S6>/Discrete motor model' */
  rtb_SpeedLimiter = 0.04877 *
    Model_2_EXPORT_Discrete_DW->Discretemotormodel_states;

  /* RateLimiter: '<S6>/Speed Limiter' */
  TrigonometricFunction = rtb_SpeedLimiter - Model_2_EXPORT_Discrete_DW->PrevY_d;
  if (TrigonometricFunction > 0.17170000000000002) {
    rtb_SpeedLimiter = Model_2_EXPORT_Discrete_DW->PrevY_d + 0.17170000000000002;
  } else {
    if (TrigonometricFunction < -0.17170000000000002) {
      rtb_SpeedLimiter = Model_2_EXPORT_Discrete_DW->PrevY_d +
        -0.17170000000000002;
    }
  }

  Model_2_EXPORT_Discrete_DW->PrevY_d = rtb_SpeedLimiter;

  /* End of RateLimiter: '<S6>/Speed Limiter' */

  /* Switch: '<S39>/Switch3' incorporates:
   *  Logic: '<S39>/ 1'
   *  Logic: '<S39>/Logical Operator'
   *  Logic: '<S39>/Logical Operator1'
   *  Logic: '<S39>/Logical Operator2'
   *  Memory: '<S39>/G3'
   *  RelationalOperator: '<S39>/Relational Operator'
   *  RelationalOperator: '<S39>/Relational Operator1'
   */
  if ((Model_2_EXPORT_Discrete_DW->G3_PreviousInput && (0.27925268031909273 <
        rtb_Sum2_e)) || ((rtb_Sum2_e <= 0.10471975511965978) &&
                         (!Model_2_EXPORT_Discrete_DW->G3_PreviousInput))) {
    /* Update for Memory: '<S39>/G3' incorporates:
     *  Logic: '<S39>/ '
     */
    Model_2_EXPORT_Discrete_DW->G3_PreviousInput =
      !Model_2_EXPORT_Discrete_DW->G3_PreviousInput;
  }

  /* End of Switch: '<S39>/Switch3' */

  /* Switch: '<S11>/Init' incorporates:
   *  Gain: '<S9>/Gain1'
   *  Inport: '<Root>/hdg0'
   *  UnitDelay: '<S11>/FixPt Unit Delay1'
   *  UnitDelay: '<S11>/FixPt Unit Delay2'
   */
  if (Model_2_EXPORT_Discrete_DW->FixPtUnitDelay2_DSTATE != 0) {
    TrigonometricFunction = 0.017453292519943295 *
      Model_2_EXPORT_Discrete_U_hdg0;
  } else {
    TrigonometricFunction = Model_2_EXPORT_Discrete_DW->FixPtUnitDelay1_DSTATE;
  }

  /* End of Switch: '<S11>/Init' */

  /* Gain: '<Root>/Gain' incorporates:
   *  Constant: '<S10>/Constant1'
   *  Gain: '<S5>/Gain'
   *  Inport: '<Root>/rudderpos'
   *  Math: '<S10>/Math Function'
   *  Quantizer: '<S1>/Quantizer2'
   *  Sum: '<Root>/Sum'
   */
  rtb_Sum2_e = (Model_2_EXPORT_Discrete_U_rudderpos - rt_roundd(rt_modd
    (TrigonometricFunction, 6.2831853071795862) / 0.0001) * 0.0001 *
                57.295779513082323) * 3.0;

  /* Update for UnitDelay: '<S8>/incidence measure' incorporates:
   *  Fcn: '<S42>/x->theta'
   *  Gain: '<S8>/V2measure'
   *  SignalConversion: '<S8>/TmpSignal ConversionAtFlipInport1'
   */
  Model_2_EXPORT_Discrete_DW->incidencemeasure_DSTATE = atan2(-rtb_Vely,
    -rtb_VAy);

  /* Update for UnitDelay: '<S11>/FixPt Unit Delay2' incorporates:
   *  Constant: '<S11>/FixPt Constant'
   */
  Model_2_EXPORT_Discrete_DW->FixPtUnitDelay2_DSTATE = 0U;

  /* Update for UnitDelay: '<S11>/FixPt Unit Delay1' incorporates:
   *  DiscreteIntegrator: '<S27>/Discrete-Time Integrator'
   */
  Model_2_EXPORT_Discrete_DW->FixPtUnitDelay1_DSTATE =
    Model_2_EXPORT_Discrete_DW->DiscreteTimeIntegrator_DSTATE;

  /* Update for DiscreteIntegrator: '<S4>/Discrete-Time Integrator1' incorporates:
   *  Gain: '<S4>/Gain1'
   *  Product: '<S33>/Product1'
   *  Sum: '<S33>/Sum'
   *  Sum: '<S4>/Sum3'
   */
  Model_2_EXPORT_Discrete_DW->DiscreteTimeIntegrator1_DSTATE += (((0.0 -
    rtb_Product_p) - rtb_Product1_d * rtb_V2measure_idx_1) + rtb_Sum_i) * 0.01 *
    0.01;

  /* Update for DiscreteIntegrator: '<S27>/Discrete-Time Integrator' */
  Model_2_EXPORT_Discrete_DW->DiscreteTimeIntegrator_IC_LOADI = 0U;

  /* Signum: '<S27>/Sign' */
  if (rtb_Velx < 0.0) {
    rtb_Velx = -1.0;
  } else {
    if (rtb_Velx > 0.0) {
      rtb_Velx = 1.0;
    }
  }

  /* End of Signum: '<S27>/Sign' */

  /* Update for DiscreteIntegrator: '<S27>/Discrete-Time Integrator' incorporates:
   *  Gain: '<S27>/Gain'
   *  Gain: '<S40>/Gain1'
   *  Product: '<S27>/Product'
   */
  Model_2_EXPORT_Discrete_DW->DiscreteTimeIntegrator_DSTATE +=
    0.017453292519943295 * rtb_SpeedLimiter * rtb_Flip_b_idx_1 * rtb_Velx * 0.3 *
    0.01;

  /* Update for DiscreteIntegrator: '<S4>/Discrete-Time Integrator' incorporates:
   *  Gain: '<S4>/Gain2'
   *  Product: '<S32>/Product2'
   *  Product: '<S32>/Product3'
   *  Product: '<S33>/Product2'
   *  Sum: '<S32>/Sum1'
   *  Sum: '<S33>/Sum1'
   *  Sum: '<S4>/Sum4'
   */
  Model_2_EXPORT_Discrete_DW->DiscreteTimeIntegrator_IC_LOA_m = 0U;
  Model_2_EXPORT_Discrete_DW->DiscreteTimeIntegrator_DSTATE_p +=
    ((rtb_Flip_idx_1 * rtb_Product3_f - rtb_Flip_idx_0 *
      rtb_TrigonometricFunction) + (rtb_Fl * rtb_V2measure_idx_1 -
      rtb_Product3_k)) * 0.033333333333333333 * 0.01;

  /* Update for DiscreteFilter: '<S4>/Discrete Filter' incorporates:
   *  Gain: '<S4>/Gain'
   *  Trigonometry: '<S4>/Trigonometric Function'
   */
  Model_2_EXPORT_Discrete_DW->DiscreteFilter_states = atan(0.010193679918450561 *
    rtb_Sum_i) - -0.99 * Model_2_EXPORT_Discrete_DW->DiscreteFilter_states;

  /* Saturate: '<S7>/Saturation' incorporates:
   *  Inport: '<Root>/sailpos'
   */
  if (Model_2_EXPORT_Discrete_U_sailpos > 13320.0) {
    TrigonometricFunction = 13320.0;
  } else if (Model_2_EXPORT_Discrete_U_sailpos < -13320.0) {
    TrigonometricFunction = -13320.0;
  } else {
    TrigonometricFunction = Model_2_EXPORT_Discrete_U_sailpos;
  }

  /* End of Saturate: '<S7>/Saturation' */

  /* Update for DiscreteFilter: '<S7>/Discrete motor model1' */
  Model_2_EXPORT_Discrete_DW->Discretemotormodel1_states = TrigonometricFunction
    - 0.9512 * Model_2_EXPORT_Discrete_DW->Discretemotormodel1_states;

  /* Saturate: '<S6>/Saturation' */
  if (rtb_Sum2_e > 13320.0) {
    rtb_Sum2_e = 13320.0;
  } else {
    if (rtb_Sum2_e < -13320.0) {
      rtb_Sum2_e = -13320.0;
    }
  }

  /* End of Saturate: '<S6>/Saturation' */

  /* Update for DiscreteFilter: '<S6>/Discrete motor model' */
  Model_2_EXPORT_Discrete_DW->Discretemotormodel_states = rtb_Sum2_e - 0.9512 *
    Model_2_EXPORT_Discrete_DW->Discretemotormodel_states;
}

/* Model initialize function */
void Model_2_EXPORT_Discrete_initialize(RT_MODEL_Model_2_EXPORT_Discr_T *const
  Model_2_EXPORT_Discrete_M, real_T *Model_2_EXPORT_Discrete_U_sailpos, real_T
  *Model_2_EXPORT_Discrete_U_rudderpos, real_T
  *Model_2_EXPORT_Discrete_U_truewindspeed, real_T
  *Model_2_EXPORT_Discrete_U_truewindheading, real_T
  *Model_2_EXPORT_Discrete_U_truewaterspeed, real_T
  *Model_2_EXPORT_Discrete_U_truewaterheading, real_T
  *Model_2_EXPORT_Discrete_U_hdg0, real_T *Model_2_EXPORT_Discrete_U_speed0,
  real_T *Model_2_EXPORT_Discrete_Y_Windincidence, real_T
  *Model_2_EXPORT_Discrete_Y_SpeedOverGround)
{
  DW_Model_2_EXPORT_Discrete_T *Model_2_EXPORT_Discrete_DW =
    ((DW_Model_2_EXPORT_Discrete_T *) Model_2_EXPORT_Discrete_M->dwork);

  /* Registration code */

  /* states (dwork) */
  (void) memset((void *)Model_2_EXPORT_Discrete_DW, 0,
                sizeof(DW_Model_2_EXPORT_Discrete_T));

  /* external inputs */
  *Model_2_EXPORT_Discrete_U_sailpos = 0.0;
  *Model_2_EXPORT_Discrete_U_rudderpos = 0.0;
  *Model_2_EXPORT_Discrete_U_truewindspeed = 0.0;
  *Model_2_EXPORT_Discrete_U_truewindheading = 0.0;
  *Model_2_EXPORT_Discrete_U_truewaterspeed = 0.0;
  *Model_2_EXPORT_Discrete_U_truewaterheading = 0.0;
  *Model_2_EXPORT_Discrete_U_hdg0 = 0.0;
  *Model_2_EXPORT_Discrete_U_speed0 = 0.0;

  /* external outputs */
  (*Model_2_EXPORT_Discrete_Y_Windincidence) = 0.0;
  (*Model_2_EXPORT_Discrete_Y_SpeedOverGround) = 0.0;

  /* InitializeConditions for UnitDelay: '<S11>/FixPt Unit Delay2' */
  Model_2_EXPORT_Discrete_DW->FixPtUnitDelay2_DSTATE = 1U;

  /* InitializeConditions for DiscreteIntegrator: '<S27>/Discrete-Time Integrator' */
  Model_2_EXPORT_Discrete_DW->DiscreteTimeIntegrator_IC_LOADI = 1U;

  /* InitializeConditions for DiscreteIntegrator: '<S4>/Discrete-Time Integrator' */
  Model_2_EXPORT_Discrete_DW->DiscreteTimeIntegrator_IC_LOA_m = 1U;

  /* InitializeConditions for RateLimiter: '<S7>/Speed Limiter' */
  Model_2_EXPORT_Discrete_DW->PrevY = 0.0;

  /* InitializeConditions for Memory: '<S39>/G3' */
  Model_2_EXPORT_Discrete_DW->G3_PreviousInput = true;

  /* InitializeConditions for RateLimiter: '<S6>/Speed Limiter' */
  Model_2_EXPORT_Discrete_DW->PrevY_d = 0.0;
}

/* Model terminate function */
void Model_2_EXPORT_Discrete_terminate(RT_MODEL_Model_2_EXPORT_Discr_T *const
  Model_2_EXPORT_Discrete_M)
{
  /* (no terminate code required) */
  UNUSED_PARAMETER(Model_2_EXPORT_Discrete_M);
}

/*
 * File trailer for generated code.
 *
 * [EOF]
 */
