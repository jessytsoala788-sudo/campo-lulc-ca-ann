// =====================================================================
// CONTROLE INDEPENDANT DES CARTES LULC DE CAMPO (reponse aux relecteurs)
// Version adaptee aux images importees dans le panneau "Imports".
// ---------------------------------------------------------------------
// AVANT DE LANCER : dans le panneau Imports (en haut de l'editeur),
// cliquez sur le nom de chaque import et renommez-le exactement ainsi :
//   - la commune (contour)        -> aoi
//   - la carte LULC de 2000       -> LULC_2000
//   - la carte LULC de 2015       -> LULC_2015
//   - la carte LULC de 2025       -> LULC_2025
// (Si vous preferez garder vos noms actuels, remplacez simplement
//  LULC_2000, LULC_2015 et LULC_2025 dans les 3 lignes ci-dessous.)
// Codes LULC : 1 FD, 2 FDg, 3 Agri, 4 Mangrove, 5 Eau, 6 Sol nu/bati
// =====================================================================

// ---------- 0. Liaison avec les images importees ------------------------
var lulc2000 = ee.Image(LULC_2000).select(0);
var lulc2015 = ee.Image(LULC_2015).select(0);
var lulc2025 = ee.Image(LULC_2025).select(0);
var zone = ee.FeatureCollection(aoi).geometry();   // fonctionne que aoi soit une geometrie ou une table

var ha = ee.Image.pixelArea().divide(1e4);

// Verification rapide : les trois cartes doivent contenir les codes 1 a 6
print('Controle 2000 (ha par classe LULC) :', ha.addBands(lulc2000).reduceRegion({
  reducer: ee.Reducer.sum().group({groupField: 1, groupName: 'classe_LULC'}),
  geometry: zone, scale: 30, maxPixels: 1e13}));

// ---------- 1. SRTM : altitude minimale et maximale --------------------
var srtm = ee.Image('USGS/SRTMGL1_003');
print('Altitude min/max (m) :', srtm.reduceRegion({
  reducer: ee.Reducer.minMax(), geometry: zone, scale: 30, maxPixels: 1e13}));

// ---------- 2. JRC Tropical Moist Forest (Vancutsem et al., 2021) -------
// Si erreur "not found" : chercher "Tropical Moist Forest" dans la barre de
// recherche GEE et remplacer v1_2023 par la version proposee (ex. v1_2024).
// Classes des bandes DecYYYY : 1 foret intacte, 2 foret degradee,
// 3 deforeste, 4 regeneration, 5 eau, 6 autres.
var tmf = ee.ImageCollection('projects/JRC/TMF/v1_2023/AnnualChanges').mosaic();

function surfacesTMF(bande, masque) {
  var img = masque ? ha.updateMask(masque) : ha;
  return img.addBands(tmf.select(bande)).reduceRegion({
    reducer: ee.Reducer.sum().group({groupField: 1, groupName: 'classe_TMF'}),
    geometry: zone, scale: 30, maxPixels: 1e13});
}
print('TMF Dec2000 (ha par classe) :', surfacesTMF('Dec2000'));
print('TMF Dec2015 (ha par classe) :', surfacesTMF('Dec2015'));
print('TMF Dec2023 (ha par classe) :', surfacesTMF('Dec2023'));

// ---------- 3. Croisement avec vos transitions ---------------------------
// a) FD (2000) -> FDg (2015) : etat TMF en 2015 ?
print('a) FD->FDg 2000-2015 : etat TMF Dec2015 (ha) :',
  surfacesTMF('Dec2015', lulc2000.eq(1).and(lulc2015.eq(2))));
// b) FDg (2015) -> FD (2025) : regeneration selon TMF ?
print('b) FDg->FD 2015-2025 : etat TMF Dec2023 (ha) :',
  surfacesTMF('Dec2023', lulc2015.eq(2).and(lulc2025.eq(1))));
// c) -> Agriculture (2015-2025) : deforeste selon TMF ?
print('c) ->Agri 2015-2025 : etat TMF Dec2023 (ha) :',
  surfacesTMF('Dec2023', lulc2015.neq(3).and(lulc2025.eq(3))));
// d) Pixels restes FD de 2000 a 2015 : intacts selon TMF ? (controle)
print('d) FD stable 2000-2015 : etat TMF Dec2015 (ha) :',
  surfacesTMF('Dec2015', lulc2000.eq(1).and(lulc2015.eq(1))));

// ---------- 4. Perte forestiere Hansen GFC par annee --------------------
// Si erreur : chercher "Hansen Global Forest Change" et prendre la version la plus recente.
var gfc = ee.Image('UMD/hansen/global_forest_change_2024_v1_12');
print('Hansen : perte (ha) par annee (1 = 2001 ... 24 = 2024) :',
  ha.updateMask(gfc.select('loss')).addBands(gfc.select('lossyear')).reduceRegion({
    reducer: ee.Reducer.sum().group({groupField: 1, groupName: 'annee'}),
    geometry: zone, scale: 30, maxPixels: 1e13}));
