%% Canon à bille à flywheels — Dimensionnement couple et anti-patinage
% Version Live Script (sections, affichages et tracés)
% Hypothèses principales :
% - Deux roues identiques (pinch), rayon r_roue
% - Rapport n = ω_moteur / ω_roue
% - Vitesse de bouche v0 estimée par ton modèle (kair linéaire)
% - Couple demandé = couple contact (accélération bille) + (option) inertiel
% - Pertes mécaniques agrégées via un rendement η

%% Paramètres de base (éditables)
n         = 2;             % [-] rapport w_mot / w_roue
mvol_pla  = 1250;          % [kg/m^3] densité plastique (bille + roue, approx.)
d_bille   = 0.006;         % [m] diamètre bille
r_roue    = 0.02;          % [m] rayon roue
h_roue    = d_bille - 0.002; % [m] "épaisseur" matière roue utilisée pour volume
eta       = 0.95;          % [-] rendement mécanique global (transmission+roulements)

% Paramètres balistique/visée
kair      = 1.02e-6;       % [N·s/m] traînée approx. linéaire
x         = 1.0;           % [m] distance d'impact
g         = 9.81;          % [m/s^2]
z0        = 0.12;          % [m] hauteur de sortie canon
z         = 0.075;         % [m] hauteur d'impact

% Commande de vitesse pendant le contact (accélération des roues)
wpt_roue  = 0;             % [rad/s^2] 0 = maintien vitesse (pas de remontée pendant t_c)

% Anti-patinage (contact bille-roues)
mu        = 0.8;           % [-] coefficient de friction gomme/plastique
k_eff     = 5000;          % [N/m] raideur équivalente de la pince (2 roues + structure)
delta     = 1.0e-3;        % [m] compression géométrique (écrasement sur le diamètre)
N_pre     = 5.0;           % [N] précharge par roue (ressort/offset). Laisser 0 si inconnue.

%% Dérivés géométriques / masses / inerties
v_roue  = pi * r_roue^2 * h_roue;                % [m^3]
v_bille = (4/3) * pi * (d_bille/2)^3;            % [m^3]

m_bille = mvol_pla * v_bille;                    % [kg]
m_roue  = mvol_pla * v_roue;                     % [kg]

J_roue  = 0.5 * m_roue * r_roue^2;               % [kg·m^2] inertie d'une roue
J_mot   = (2 * J_roue) * n^2;                    % [kg·m^2] inertie reflétée au moteur (2 roues, x n^2)

%% Vitesse de bouche et vitesses de rotation
v0 = (kair/m_bille) * ( x / (1 - exp( -(kair/m_bille) * sqrt((2*(z0 - z))/g) )) );  % [m/s]

w_roue = v0 / r_roue;     % [rad/s]
w_mot  = n * w_roue;      % [rad/s]

%% Temps de contact (approx) et effort tangent
t_c    = d_bille / max(v0, 1e-9);      % [s] ~ temps pour parcourir un diamètre dans la pince
F_t    = m_bille * (v0 / t_c);         % [N] force tangentielle pour accélérer la bille
tau_c  = r_roue * F_t;                 % [N·m] couple de contact total (sur l'arbre roue)

%% Terme inertiel moteur (optionnel via wpt_roue)
wpt_mot   = n * wpt_roue;              % [rad/s^2]
tau_inert = J_mot * wpt_mot;           % [N·m] couple inertiel au moteur

%% Couple moteur requis (avec rendement)
% (On ne modélise pas finement les pertes visqueuses/aéro ici — incluses via η)
T_mot = (tau_c + tau_inert) / eta;     % [N·m]

%% Anti-patinage : marge de friction
% Force normale par roue : N = k_eff*delta + N_pre
N_par_roue = k_eff * delta + N_pre;    % [N]
F_max      = 2 * mu * N_par_roue;      % [N] total (deux roues)
marge      = F_max / max(F_t, 1e-9);   % [-] >1 : pas de patinage, <1 : patine

%% Affichages
fprintf('--- Résultats principaux ---\n');
fprintf('m_bille        = %.6f kg\n', m_bille);
fprintf('m_roue         = %.6f kg\n', m_roue);
fprintf('J_roue         = %.3e kg·m^2\n', J_roue);
fprintf('J_moteur(eq)   = %.3e kg·m^2\n', J_mot);
fprintf('v0             = %.3f m/s\n', v0);
fprintf('w_roue         = %.1f rad/s (%.0f rpm)\n', w_roue, rad2deg(w_roue)*60/360);
fprintf('w_mot          = %.1f rad/s (%.0f rpm)\n', w_mot, rad2deg(w_mot)*60/360);
fprintf('t_c (contact)  = %.6f s\n', t_c);
fprintf('F_t (requis)   = %.3f N\n', F_t);
fprintf('tau_contact    = %.5f N·m\n', tau_c);
fprintf('tau_inertiel   = %.5f N·m (wpt_roue=%.1f rad/s^2)\n', tau_inert, wpt_roue);
fprintf('T_mot (total)  = %.5f N·m (avec eta=%.2f)\n', T_mot, eta);
fprintf('Anti-patinage: N/roue=%.2f N, F_max=%.2f N, marge=%.2f\n', ...
        N_par_roue, F_max, marge);

%% Tracé 1 — Influence du rapport n sur le couple requis
n_vec = linspace(1, 5, 81);
Jmot_vec   = (2 * J_roue) .* (n_vec.^2);
wpt_mot_v  = n_vec * wpt_roue;
Tmot_vec   = (tau_c + Jmot_vec .* wpt_mot_v) / eta;

figure; %#ok<UNRCH>
plot(n_vec, Tmot_vec, 'LineWidth', 1.5);
grid on;
xlabel('Rapport n = \omega_{mot}/\omega_{roue} [-]');
ylabel('Couple moteur requis T_{mot} [N·m]');
title('Influence du rapport n sur le couple requis');

%% Tracé 2 — Marge d'adhérence vs précharge
Npre_vec = linspace(0, 20, 101);     % [N] par roue
marge_vec = (2 * mu * (k_eff*delta + Npre_vec)) / max(F_t, 1e-9);

figure;
plot(Npre_vec, marge_vec, 'LineWidth', 1.5);
yline(1,'--'); grid on;
xlabel('Précharge par roue N_{pre} [N]');
ylabel('Marge d''adhérence (F_{max}/F_t) [-]');
title('Marge d adherence en fonction de la precharge');

%% (Option) Fonction utilitaire : calcul compact regroupé
% Exemple d'usage :
% res = calc_flywheel(d_bille, r_roue, h_roue, mvol_pla, n, kair, x, z0, z, g, ...
%                     eta, wpt_roue, mu, k_eff, delta, N_pre);
% disp(res)
function out = calc_flywheel(d_bille, r_roue, h_roue, rho, n, kair, x, z0, z, g, eta, wpt_roue, mu, k_eff, delta, N_pre)
    v_roue  = pi * r_roue^2 * h_roue;
    v_bille = (4/3) * pi * (d_bille/2)^3;
    m_bille = rho * v_bille;
    m_roue  = rho * v_roue;
    J_roue  = 0.5 * m_roue * r_roue^2;
    J_mot   = (2 * J_roue) * n^2;
    v0      = (kair/m_bille) * ( x / (1 - exp( -(kair/m_bille) * sqrt((2*(z0 - z))/g) )) );
    w_roue  = v0 / r_roue;
    w_mot   = n * w_roue;
    t_c     = d_bille / max(v0, 1e-9);
    F_t     = m_bille * (v0 / t_c);
    tau_c   = r_roue * F_t;
    wpt_mot = n * wpt_roue;
    tau_in  = J_mot * wpt_mot;
    T_mot   = (tau_c + tau_in) / eta;
    N_roue  = k_eff * delta + N_pre;
    F_max   = 2 * mu * N_roue;
    marge   = F_max / max(F_t,1e-9);
    out = struct('m_bille',m_bille,'m_roue',m_roue,'J_roue',J_roue,'J_mot',J_mot, ...
                 'v0',v0,'w_roue',w_roue,'w_mot',w_mot,'t_c',t_c, ...
                 'F_t',F_t,'tau_contact',tau_c,'tau_inertiel',tau_in, ...
                 'T_mot',T_mot,'N_par_roue',N_roue,'F_max',F_max,'marge',marge);
end
