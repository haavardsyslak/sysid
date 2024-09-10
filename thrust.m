
T = [0.70710678 -0.43297191  0.43153573 -0.70710678  0 0 0 0;
    -0.70710678 -0.90140741 0.90209584 0.70710678 0 0 0 0;
    0 0 0 0 -1 -1 -1 -1;
    -0.06010408 -0.07661963 0.07667815 0.06010408 0.218 -0.21775755 0.21756546 -0.218;
    -0.06010408 0.03680261 -0.03668054 0.06010408 -0.12 -0.1204394 0.12078606  0.12;
    0.18879751 0.18879751 0.18879751 0.18879751 0 -0 0 0];




k = tf([6136, 108700], [1, 89, 9258, 108700]);
K = eye(8) * k

t = 0:0.01:15;

u = zeros([8, length(t)]);

u(5:8, t >= 5 & t <= 10) = 1; 
u = transpose(u)
thrust_u = arrayfun(@(v) get_thrust(v), u);


%  how to get the time domain representation of tau?


function thrust = get_thrust(V)
    thrust = -140.3 * V.^9 + 389.9 * V.^7 - 404.1 * V.^5 + 176.0 * V.^3 + 8.9 * V;
end
