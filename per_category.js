// One pairwise A/B evaluation question for each video category.
const CATEGORY_CODE_MAP = {
  FA: "falling",
  PR: "projectile",
  SW: "swinging",
  CP: "compression",
  SP: "spinning",
  RO: "sliding",
  BO: "bouncing",
  FL: "fluid flowing",
  FT: "floating",
  SL: "splashing",
};

const CATEGORY_PAIRWISE_QUESTIONS = Object.fromEntries(
  Object.entries(CATEGORY_CODE_MAP).map(([code, category]) => [
    code,
    `Which video generates a more natural physical ${category} behavior while also aligning closely with the text description? You must select Video A or Video B; ties are not allowed.`,
  ]),
);

module.exports = {
  CATEGORY_CODE_MAP,
  CATEGORY_PAIRWISE_QUESTIONS,
};
