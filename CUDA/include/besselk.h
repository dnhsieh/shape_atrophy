#ifndef BESSELK_H
#define BESSELK_H

const int    P01Deg = 5;
const double P01Vec[6] = { 2.4708152720399552679e+03,  5.9169059852270512312e+03,
                           4.6850901201934832188e+02,  1.1999463724910714109e+01,
                           1.3166052564989571850e-01,  5.8599221412826100000e-04};
__constant__ double c_P01Vec[6];

const int    dP01Deg = 4;
const double dP01Vec[5] = {                                 1 * 5.9169059852270512312e+03,
                            2 * 4.6850901201934832188e+02,  3 * 1.1999463724910714109e+01,
                            4 * 1.3166052564989571850e-01,  5 * 5.8599221412826100000e-04};
__constant__ double c_dP01Vec[5];

// ---

const int    Q01Deg = 2;
const double Q01Vec[3] = { 2.1312714303849120380e+04, -2.4994418972832303646e+02,
                           1                                                    };
__constant__ double c_Q01Vec[3];

const int    dQ01Deg = 1;
const double dQ01Vec[2] = {                                -1 * 2.4994418972832303646e+02,
                            2 * 1                                                        };
__constant__ double c_dQ01Vec[2];

// ---

const int    P02Deg = 3;
const double P02Vec[4] = {-4.0320340761145482298e+05, -1.7733784684952985886e+04,
                          -2.9601657892958843866e+02, -1.6414452837299064100e+00};
__constant__ double c_P02Vec[4];

const int    dP02Deg = 2;
const double dP02Vec[3] = {                                -1 * 1.7733784684952985886e+04,
                           -2 * 2.9601657892958843866e+02, -3 * 1.6414452837299064100e+00};
__constant__ double c_dP02Vec[3];

// ---

const int    Q02Deg = 3;
const double Q02Vec[4] = {-1.6128136304458193998e+06,  2.9865713163054025489e+04,
                          -2.5064972445877992730e+02,  1                        };
__constant__ double c_Q02Vec[4];

const int    dQ02Deg = 2;
const double dQ02Vec[3] = {                                 1 * 2.9865713163054025489e+04,
                           -2 * 2.5064972445877992730e+02,  3 * 1                        };
__constant__ double c_dQ02Vec[3];

// ---

const int    P03Deg = 9;
const double P03Vec[10] = { 1.1600249425076035558e+02,  2.3444738764199315021e+03,
                            1.8321525870183537725e+04,  7.1557062783764037541e+04,
                            1.5097646353289914539e+05,  1.7398867902565686251e+05,
                            1.0577068948034021957e+05,  3.1075408980684392399e+04,
                            3.6832589957340267940e+03,  1.1394980557384778174e+02};
__constant__ double c_P03Vec[10];

const int    dP03Deg = 8;
const double dP03Vec[9] = {                                 1 * 2.3444738764199315021e+03,
                            2 * 1.8321525870183537725e+04,  3 * 7.1557062783764037541e+04,
                            4 * 1.5097646353289914539e+05,  5 * 1.7398867902565686251e+05,
                            6 * 1.0577068948034021957e+05,  7 * 3.1075408980684392399e+04,
                            8 * 3.6832589957340267940e+03,  9 * 1.1394980557384778174e+02};
__constant__ double c_dP03Vec[9];

// ---

const int    Q03Deg = 10;
const double Q03Vec[11] = { 9.2556599177304839811e+01,  1.8821890840982713696e+03,
                            1.4847228371802360957e+04,  5.8824616785857027752e+04,
                            1.2689839587977598727e+05,  1.5144644673520157801e+05,
                            9.7418829762268075784e+04,  3.1474655750295278825e+04,
                            4.4329628889746408858e+03,  2.0013443064949242491e+02,
                            1                                                    };
__constant__ double c_Q03Vec[11];

const int    dQ03Deg = 9;
const double dQ03Vec[10] = {                                 1 * 1.8821890840982713696e+03,
                             2 * 1.4847228371802360957e+04,  3 * 5.8824616785857027752e+04,
                             4 * 1.2689839587977598727e+05,  5 * 1.5144644673520157801e+05,
                             6 * 9.7418829762268075784e+04,  7 * 3.1474655750295278825e+04,
                             8 * 4.4329628889746408858e+03,  9 * 2.0013443064949242491e+02,
                            10 * 1                                                        };
__constant__ double c_dQ03Vec[10];

// ---

const int    P11Deg = 5;
const double P11Vec[6] = {-2.2149374878243304548e+06,  7.1938920065420586101e+05,
                           1.7733324035147015630e+05,  7.1885382604084798576e+03,
                           9.9991373567429309922e+01,  4.8127070456878442310e-01};
__constant__ double c_P11Vec[6];

const int    dP11Deg = 4;
const double dP11Vec[5] = {                                 1 * 7.1938920065420586101e+05,
                            2 * 1.7733324035147015630e+05,  3 * 7.1885382604084798576e+03,
                            4 * 9.9991373567429309922e+01,  5 * 4.8127070456878442310e-01};
__constant__ double c_dP11Vec[5];

// ---

const int    Q11Deg = 3;
const double Q11Vec[4] = {-2.2149374878243304548e+06,  3.7264298672067697862e+04,
                          -2.8143915754538725829e+02,  1                        };
__constant__ double c_Q11Vec[4];

const int    dQ11Deg = 2;
const double dQ11Vec[3] = {                                 1 * 3.7264298672067697862e+04,
                           -2 * 2.8143915754538725829e+02,  3 * 1                        };
__constant__ double c_dQ11Vec[3];

// ---

const int    P12Deg = 4;
const double P12Vec[5] = {-1.3531161492785421328e+06, -1.4758069205414222471e+05,
                          -4.5051623763436087023e+03, -5.3103913335180275253e+01,
                          -2.2795590826955002390e-01                            };
__constant__ double c_P12Vec[5];

const int    dP12Deg = 3;
const double dP12Vec[4] = {                                -1 * 1.4758069205414222471e+05,
                           -2 * 4.5051623763436087023e+03, -3 * 5.3103913335180275253e+01,
                           -4 * 2.2795590826955002390e-01                                };
__constant__ double c_dP12Vec[4];

// ---

const int    Q12Deg = 3;
const double Q12Vec[4] = {-2.7062322985570842656e+06,  4.3117653211351080007e+04,
                          -3.0507151578787595807e+02,  1                        };
__constant__ double c_Q12Vec[4];

const int    dQ12Deg = 2;
const double dQ12Vec[3] = {                                 1 * 4.3117653211351080007e+04,
                           -2 * 3.0507151578787595807e+02,  3 * 1                        };
__constant__ double c_dQ12Vec[3];

// ---

const int    P13Deg = 10;
const double P13Vec[11] = { 2.2196792496874548962e+00,  4.4137176114230414036e+01,
                            3.4122953486801312910e+02,  1.3319486433183221990e+03,
                            2.8590657697910288226e+03,  3.4540675585544584407e+03,
                            2.3123742209168871550e+03,  8.1094256146537402173e+02,
                            1.3182609918569941308e+02,  7.5584584631176030810e+00,
                            6.4257745859173138767e-02                            };
__constant__ double c_P13Vec[11];

const int    dP13Deg = 9;
const double dP13Vec[10] = {                                 1 * 4.4137176114230414036e+01,
                             2 * 3.4122953486801312910e+02,  3 * 1.3319486433183221990e+03,
                             4 * 2.8590657697910288226e+03,  5 * 3.4540675585544584407e+03,
                             6 * 2.3123742209168871550e+03,  7 * 8.1094256146537402173e+02,
                             8 * 1.3182609918569941308e+02,  9 * 7.5584584631176030810e+00,
                            10 * 6.4257745859173138767e-02                                };
__constant__ double c_dP13Vec[10];

// ---

const int    Q13Deg = 9;
const double Q13Vec[10] = { 1.7710478032601086579e+00,  3.4552228452758912848e+01,
                            2.5951223655579051357e+02,  9.6929165726802648634e+02,
                            1.9448440788918006154e+03,  2.1181000487171943810e+03,
                            1.2082692316002348638e+03,  3.3031020088765390854e+02,
                            3.6001069306861518855e+01,  1                        };
__constant__ double c_Q13Vec[10];

const int    dQ13Deg = 8;
const double dQ13Vec[9] = {                                 1 * 3.4552228452758912848e+01,
                            2 * 2.5951223655579051357e+02,  3 * 9.6929165726802648634e+02,
                            4 * 1.9448440788918006154e+03,  5 * 2.1181000487171943810e+03,
                            6 * 1.2082692316002348638e+03,  7 * 3.3031020088765390854e+02,
                            8 * 3.6001069306861518855e+01,  9 * 1                        };
__constant__ double c_dQ13Vec[9];

// ---

const double xMax = 700.0;

#endif
