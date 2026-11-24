%% Density

DENSITY_Pred_Try_02 = readtable("D:\THESIS_WORK_FINAL\FINAL CODES\SOD_SHOCK_TUBE\PINNS\Comparision_Plot\Try_3\t_02\DENSITY_Pred_Try_02.csv");
Exact_Density_02 = readtable("D:\THESIS_WORK_FINAL\FINAL CODES\SOD_SHOCK_TUBE\PINNS\Comparision_Plot\Try_3\t_02\Exact_Density_02.csv");
Test_data_Try_02 = readtable("D:\THESIS_WORK_FINAL\FINAL CODES\SOD_SHOCK_TUBE\PINNS\Comparision_Plot\Try_3\t_02\Test_data_Try_02.csv");
XT_EXACT_02 = readtable("D:\THESIS_WORK_FINAL\FINAL CODES\SOD_SHOCK_TUBE\PINNS\Comparision_Plot\Try_3\t_02\XT_EXACT_02.csv");

DENSITY_Pred_Try_02=table2array(DENSITY_Pred_Try_02);
Exact_Density_02=table2array(Exact_Density_02);
Test_data_Try_02=table2array(Test_data_Try_02);
XT_EXACT_02=table2array(XT_EXACT_02);

plot(Test_data_Try_02,DENSITY_Pred_Try_02,color="r");
hold on;
plot(XT_EXACT_02,Exact_Density_02,color='b');
axis square;
xlabel("x")
ylabel("\rho(x,t=0.2)")
title("Density at t=0.2 s")

%% Pressure

Pressure_Pred_Try_02 = readtable("D:\THESIS_WORK_FINAL\FINAL CODES\SOD_SHOCK_TUBE\PINNS\Comparision_Plot\Try_3\t_02\Pressure_Pred_Try_02.csv");
Exact_Pressure_02 = readtable("D:\THESIS_WORK_FINAL\FINAL CODES\SOD_SHOCK_TUBE\PINNS\Comparision_Plot\Try_3\t_02\Exact_Pressure_02.csv");
Test_data_Try_02 = readtable("D:\THESIS_WORK_FINAL\FINAL CODES\SOD_SHOCK_TUBE\PINNS\Comparision_Plot\Try_3\t_02\Test_data_Try_02.csv");
XT_EXACT_02 = readtable("D:\THESIS_WORK_FINAL\FINAL CODES\SOD_SHOCK_TUBE\PINNS\Comparision_Plot\Try_3\t_02\XT_EXACT_02.csv");

Pressure_Pred_Try_02=table2array(Pressure_Pred_Try_02);
Exact_Pressure_02=table2array(Exact_Pressure_02);
Test_data_Try_02=table2array(Test_data_Try_02);
XT_EXACT_02=table2array(XT_EXACT_02);

plot(Test_data_Try_02,Pressure_Pred_Try_02,color="r");
hold on;
plot(XT_EXACT_02,Exact_Pressure_02,color='b');
axis square;
xlabel("x")
ylabel("p(x,t=0.2)")
title("Pressure at t=0.2 s")

%% Velocity

Velocity_Pred_Try_02 = readtable("D:\THESIS_WORK_FINAL\FINAL CODES\SOD_SHOCK_TUBE\PINNS\Comparision_Plot\Try_3\t_02\Velocity_Pred_Try_02.csv");
Exact_Velocity_02 = readtable("D:\THESIS_WORK_FINAL\FINAL CODES\SOD_SHOCK_TUBE\PINNS\Comparision_Plot\Try_3\t_02\Exact_Velocity_02.csv");
XT_EXACT_02 = readtable("D:\THESIS_WORK_FINAL\FINAL CODES\SOD_SHOCK_TUBE\PINNS\Comparision_Plot\Try_3\t_02\XT_EXACT_02.csv");
Test_data_Try_02 = readtable("D:\THESIS_WORK_FINAL\FINAL CODES\SOD_SHOCK_TUBE\PINNS\Comparision_Plot\Try_3\t_02\Test_data_Try_02.csv");

Velocity_Pred_Try_02=table2array(Velocity_Pred_Try_02);
Exact_Velocity_02=table2array(Exact_Velocity_02);
Test_data_Try_02=table2array(Test_data_Try_02);
XT_EXACT_02=table2array(XT_EXACT_02);

plot(Test_data_Try_02,Velocity_Pred_Try_02,color="r");
hold on;
plot(XT_EXACT_02,Exact_Velocity_02,color='b');
axis square;
xlabel("x")
ylabel("v(x,t=0.2)")
title("Velcoity at t=0.2 s")
