%% ===============================================================
%% Système flywheel pour lancer une bille — Modèle énergie + droop
%% ===============================================================

clear; clc;

%% ===========================
%% Paramètres généraux
%% ===========================
n = 1;                 % rapport d'engrenage = w_mot / w_roue
eta = 0.95;            % rendement méca global
mvol_pla = 1250;       % kg/m^3 (densité PLA)
d_bille = 0.006;       % m (diamètre bille)
r_roue = 0.02;         % m (rayon roue)
h_roue = d_bille - 0.002; % m (épaisseur roue utile)

Ts = 15e-3;            % N.m (couple de décrochage moteur)
n0 = 7500;             % rpm (vitesse à vide moteur)
Jmot = 5e-7;           % kg.m^2 inertie rotor (ordre de grandeur)
eta_spin = 0.95;       % rendement pendant spin-up
dt = 6;                % s (temps de spin-up)

% Données balistiques (pour le calcul de v0)
kair = 1.02E-6;        % N.s/m (coefficient air)
x = 1;                 % distance de l'impact
g = 9.81;
z0 = 0.12;             % hauteur du canon
z = 0.075;             % hauteur d'impact

%% ===========================
%% Géométrie, volumes, masses, inerties
%% ===========================
v_roue = pi * r_roue^2 * h_roue;
v_bille = (4/3) * pi * (d_bille/2)^3;

m_bille = mvol_pla * v_bille;
m_roue  = mvol_pla * v_roue;

Jroue = 0.5 * m_roue * r_roue^2;
Jtot  = 2 * Jroue * n^2 + Jmot;    % inertie reflétée au moteur (2 roues + rotor)

%% ===========================
%% Balistique → vitesse de bouche cible
%% ===========================
v0 = (kair/m_bille) * (x / (1 - exp(-(kair/m_bille)*sqrt((2*(z0-z))/g)))); % m/s

%% ===========================
%% Cinématique roues / moteur (cible)
%% ===========================
w_roue = v0 / r_roue;           % rad/s
w_mot  = w_roue * n;

rpm_roue = (w_roue/(2*pi))*60;
rpm_mot  = rpm_roue * n;

%% ===========================
%% Phase 1 — Spin-up (0 -> w_roue en dt)
%% ===========================
w_pt_roue_spin = w_roue / dt;            % accélération angulaire roue
w_pt_mot_spin  = w_pt_roue_spin * n;     % accélération moteur
T_spin = (Jtot * w_pt_mot_spin) / eta_spin;  % N.m (couple requis pour spin-up)

%% ===========================
%% Phase 2 — Contact (modèle énergie + droop)
%% ===========================
beta = 1.5;                       % marge pertes (1.3–1.6 conseillé)
E_ball = 0.5 * m_bille * v0^2;    % énergie à communiquer à la bille
E_need = beta * E_ball;           % énergie totale à fournir par les roues

omega = w_roue;                   % vitesse angulaire roue à vide
A = Jroue * omega^2;              % énergie "réservoir" d'une roue
rhs = E_need / (2*A);             % (2s - s^2) = rhs (2 roues)
s = 1 - sqrt(max(1 - rhs, 0));    % droop relatif (baisse de vitesse des roues)
w_after = omega * (1 - s);        % vitesse après tir

%% ===========================
%% Puissance moteur et temps de recharge
%% ===========================
T_at_rpm = Ts * (1 - rpm_mot/n0);       % couple dispo à ce régime
P_mot = max(T_at_rpm, 0) * omega;       % puissance méca dispo
t_recharge = E_need / max(P_mot, 1e-6); % temps de recharge estimé

%% ===========================
%% Résumé des résultats
%% ===========================
fprintf('\n=== Résultats principaux ===\n');
fprintf('Vitesse visée v0          = %.2f m/s\n', v0);
fprintf('Masse bille               = %.4f g\n', m_bille*1000);
fprintf('Vitesse roue cible        = %.0f rpm\n', rpm_roue);
fprintf('Couple spin-up            = %.4e N.m\n', T_spin);
fprintf('Énergie bille             = %.4f mJ\n', E_ball*1000);
fprintf('Énergie fournie (avec pertes) = %.4f mJ\n', E_need*1000);
fprintf('Droop relatif (chute ω)   = %.2f %%\n', s*100);
fprintf('Vitesse roue après tir    = %.0f rpm\n', (w_after/(2*pi))*60);
fprintf('Couple moteur dispo       = %.4f N.m\n', T_at_rpm);
fprintf('Puissance méca moteur     = %.3f W\n', P_mot);
fprintf('Temps recharge estimé     = %.3f ms\n', t_recharge*1e3);

%% ===========================
%% Vérification marges moteur
%% ===========================
if T_spin > Ts
    warning('Le couple moteur est insuffisant pour le spin-up (%.3f > %.3f N.m).', T_spin, Ts);
end
if s > 0.1
    warning('Droop supérieur à 10%% (%.2f%%). Inertie ou rendement insuffisant.', s*100);
end