
pub struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    pub fn len(&self) -> usize {
        self.parent.len()
    }

    pub fn is_empty(&self) -> bool {
        self.parent.is_empty()
    }

    pub fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    pub fn union(&mut self, a: usize, b: usize) -> bool {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return false;
        }
        let (small, big) = if self.rank[ra] < self.rank[rb] {
            (ra, rb)
        } else {
            (rb, ra)
        };
        self.parent[small] = big;
        if self.rank[ra] == self.rank[rb] {
            self.rank[big] += 1;
        }
        true
    }

    pub fn components(&mut self) -> Vec<Vec<usize>> {
        use std::collections::BTreeMap;
        let mut by_root: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for i in 0..self.parent.len() {
            let r = self.find(i);
            by_root.entry(r).or_default().push(i);
        }
        by_root.into_values().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_then_all_singletons() {
        let mut uf = UnionFind::new(5);
        let comps = uf.components();
        assert_eq!(comps.len(), 5);
        for c in &comps {
            assert_eq!(c.len(), 1);
        }
    }

    #[test]
    fn union_merges_components() {
        let mut uf = UnionFind::new(4);
        assert!(uf.union(0, 1));
        assert!(uf.union(2, 3));
        assert_eq!(uf.components().len(), 2);
        assert!(uf.union(1, 2));
        assert_eq!(uf.components().len(), 1);
    }

    #[test]
    fn union_is_idempotent() {
        let mut uf = UnionFind::new(3);
        assert!(uf.union(0, 1));
        assert!(!uf.union(0, 1));
        assert!(!uf.union(1, 0));
        assert_eq!(uf.find(0), uf.find(1));
    }

    #[test]
    fn components_groups_correctly() {
        let mut uf = UnionFind::new(6);
        uf.union(0, 2);
        uf.union(2, 4);
        uf.union(1, 3);
        let mut comps = uf.components();
        for c in comps.iter_mut() {
            c.sort();
        }
        comps.sort_by_key(|c| c[0]);
        assert_eq!(comps, vec![vec![0, 2, 4], vec![1, 3], vec![5]]);
    }

    #[test]
    fn empty_is_empty() {
        let mut uf = UnionFind::new(0);
        assert!(uf.is_empty());
        assert!(uf.components().is_empty());
    }
}
