Pred_density_Entropy = readtable("D:\THESIS_WORK_FINAL\FINAL CODES\SOD_SHOCK_TUBE\PINNS\Comparision_Plot\Try_1\DENSITY_Pred_Entropy.csv");
Pressure_Pred_Entropy = readtable("D:\THESIS_WORK_FINAL\FINAL CODES\SOD_SHOCK_TUBE\PINNS\Comparision_Plot\Try_1\Pressure_Pred_Entropy.csv");
Velocity_Pred_Entropy = readtable("D:\THESIS_WORK_FINAL\FINAL CODES\SOD_SHOCK_TUBE\PINNS\Comparision_Plot\Try_1\Velocity_Pred_Entropy.csv");
Test_data_0_101_Entropy = readtable("D:\THESIS_WORK_FINAL\FINAL CODES\SOD_SHOCK_TUBE\PINNS\Comparision_Plot\Try_1\Test_data_0_101_Entropy.csv");

Pred_density_Entropy=table2array(Pred_density_Entropy);
Pressure_Pred_Entropy=table2array(Pressure_Pred_Entropy);
Velocity_Pred_Entropy=table2array(Velocity_Pred_Entropy);
Test_data_0_101_Entropy=table2array(Test_data_0_101_Entropy);


Density_Pred_W_PINNS_DE_t2 = readtable("D:\THESIS_WORK_FINAL\FINAL CODES\SOD_SHOCK_TUBE\PINNS\Comparision_Plot\Try_1\Density_Pred_W_PINNS_DE_t2.csv");
Pressure_Pred_W_PINNS_DE_t2 = readtable("D:\THESIS_WORK_FINAL\FINAL CODES\SOD_SHOCK_TUBE\PINNS\Comparision_Plot\Try_1\Pressure_Pred_W_PINNS_DE_t2.csv");
Velocity_Pred_W_PINNS_DE_t2 = readtable("D:\THESIS_WORK_FINAL\FINAL CODES\SOD_SHOCK_TUBE\PINNS\Comparision_Plot\Try_1\Velocity_Pred_W_PINNS_DE_t2.csv");
XT_TEST_W_PINNS_DE_t2 = readtable("D:\THESIS_WORK_FINAL\FINAL CODES\SOD_SHOCK_TUBE\PINNS\Comparision_Plot\Try_1\XT_TEST_W_PINNS_DE_t2.csv");

Density_Pred_W_PINNS_DE_t2=table2array(Density_Pred_W_PINNS_DE_t2);
Pressure_Pred_W_PINNS_DE_t2=table2array(Pressure_Pred_W_PINNS_DE_t2);
Velocity_Pred_W_PINNS_DE_t2=table2array(Velocity_Pred_W_PINNS_DE_t2);
XT_TEST_W_PINNS_DE_t2=table2array(XT_TEST_W_PINNS_DE_t2);

Exact_Density = readtable("D:\THESIS_WORK_FINAL\FINAL CODES\SOD_SHOCK_TUBE\PINNS\Comparision_Plot\Try_1\Exact_Density.csv");
Exact_Pressure = readtable("D:\THESIS_WORK_FINAL\FINAL CODES\SOD_SHOCK_TUBE\PINNS\Comparision_Plot\Try_1\Exact_Pressure.csv");
Exact_Velocity = readtable("D:\THESIS_WORK_FINAL\FINAL CODES\SOD_SHOCK_TUBE\PINNS\Comparision_Plot\Try_1\Exact_Velocity.csv");
XT_EXACT = readtable("D:\THESIS_WORK_FINAL\FINAL CODES\SOD_SHOCK_TUBE\PINNS\Comparision_Plot\Try_1\XT_EXACT.csv");

Exact_Density=table2array(Exact_Density);
Exact_Pressure=table2array(Exact_Pressure);
Exact_Velocity=table2array(Exact_Velocity);
XT_EXACT=table2array(XT_EXACT);

CASE1_PREDICTED_Density_02_LOW_FIDLEITY = readtable("D:\THESIS_WORK_FINAL\FINAL CODES\SOD_SHOCK_TUBE\PINNS\Comparision_Plot\Try_1\CASE1_PREDICTED_Density_02_LOW_FIDLEITY.csv");
CASE1_X_VAL_015_LOW_FIDELITY = readtable("D:\THESIS_WORK_FINAL\FINAL CODES\SOD_SHOCK_TUBE\PINNS\Comparision_Plot\Try_1\CASE1_X_VAL_015_LOW_FIDELITY.csv");
CASE1_PREDICTED_PRESSURE_02_LOW_FIDELITY = readtable("D:\THESIS_WORK_FINAL\FINAL CODES\SOD_SHOCK_TUBE\PINNS\Comparision_Plot\Try_1\CASE1_PREDICTED_PRESSURE_02_LOW_FIDELITY.csv");
%CASE1_X_VAL_02_LOW_FIDELITY = readtable("D:\THESIS_WORK_FINAL\FINAL CODES\SOD_SHOCK_TUBE\PINNS\Comparision_Plot\Try_1\CASE1_X_VAL_02_LOW_FIDELITY.csv");

CASE1_PREDICTED_Velocity_02_LOW_FIDELITY_DATA = readtable("D:\THESIS_WORK_FINAL\FINAL CODES\SOD_SHOCK_TUBE\PINNS\Comparision_Plot\Try_1\CASE1_PREDICTED_Velocity_02_LOW_FIDELITY_DATA.csv");


CASE1_PREDICTED_Velocity_02_LOW_FIDELITY_DATA=table2array(CASE1_PREDICTED_Velocity_02_LOW_FIDELITY_DATA);
CASE1_PREDICTED_Density_02_LOW_FIDLEITY=table2array(CASE1_PREDICTED_Density_02_LOW_FIDLEITY);
CASE1_X_VAL_015_LOW_FIDELITY=table2array(CASE1_X_VAL_015_LOW_FIDELITY);
CASE1_PREDICTED_PRESSURE_02_LOW_FIDELITY=table2array(CASE1_PREDICTED_PRESSURE_02_LOW_FIDELITY);
%CASE1_X_VAL_02_LOW_FIDELITY=table2array(CASE1_X_VAL_02_LOW_FIDELITY);

%%
% plot(Test_data_0_101_Entropy,Pred_density_Entropy,color='R');
% hold on;
% plot(XT_TEST_W_PINNS_DE_t2,Density_Pred_W_PINNS_DE_t2,color='g');
% hold on;
% plot(XT_EXACT,Exact_Density,'b');
% hold on;
% plot(Test_data_0_101_Entropy,CASE1_PREDICTED_Density_02_LOW_FIDLEITY,color='m')
% axis square;
%%
% plot(Test_data_0_101_Entropy,Pressure_Pred_Entropy,color='R');
% hold on;
% plot(XT_TEST_W_PINNS_DE_t2,Velocity_Pred_W_PINNS_DE_t2,color='g');
% hold on;
% plot(Test_data_0_101_Entropy,CASE1_PREDICTED_PRESSURE_02_LOW_FIDELITY,'m');
% hold on;
% plot(XT_EXACT,Exact_Pressure,color='b')
% axis square;

%%
plot(Test_data_0_101_Entropy,Velocity_Pred_Entropy,color='R');
hold on;
plot(XT_TEST_W_PINNS_DE_t2,Pressure_Pred_W_PINNS_DE_t2,color='g');
hold on;
plot(Test_data_0_101_Entropy,CASE1_PREDICTED_Velocity_02_LOW_FIDELITY_DATA,'m');
hold on;
plot(XT_EXACT,Exact_Velocity,color='b')
axis square;