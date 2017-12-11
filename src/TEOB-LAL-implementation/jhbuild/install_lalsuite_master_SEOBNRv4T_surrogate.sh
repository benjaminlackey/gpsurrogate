#!/bin/bash

# Create pe directory where the installation will be
mkdir -p ~/pe
cd ~/pe

# Download module sets for jhbuild
# Already checked out and some local modifications: lsc_mpuerrer.modules
#if [ ! -d ~/pe/modulesets ]; then
#  git clone git://github.com/vivienr/modulesets.git ~/pe/modulesets
#fi
cd ~/pe/modulesets
#git pull

# Download and install jhbuild
mkdir -p ~/pe/src
if [ ! -d ~/pe/src/jhbuild ]; then
  git clone git://git.gnome.org/jhbuild ~/pe/src/jhbuild
  cd ~/pe/src/jhbuild
  ./autogen.sh --prefix=~/pe/.local/
  make
  make install
fi

# Set up jhbuild configuration files
mkdir -p ~/pe/.config && cd ~/pe/.config
if [ ! -e ~/pe/.config/jhbuildrc ]; then
  ln -s ~/pe/modulesets/jhbuildrc
fi


echo "prefix = '$HOME/pe/local/'" > ~/pe/.config/pe.jhbuildrc
# override default from jhbuildrc
echo "moduleset = [ 'daswg', 'lsc_mpuerrer', 'github', 'sourceforge', 'gnu', 'gnome', 'bitbucket', 'nds2', 'healpix' ]" >> ~/pe/.config/pe.jhbuildrc
echo "modulesets_dir = '$HOME/pe/modulesets'" >> ~/pe/.config/pe.jhbuildrc
echo "checkoutroot = '$HOME/pe/src'" >> ~/pe/.config/pe.jhbuildrc
echo "branches['lal'] = (None,'master')" >> ~/pe/.config/pe.jhbuildrc
echo "branches['laldetchar'] = (None,'master')" >> ~/pe/.config/pe.jhbuildrc
echo "branches['lalframe'] = (None,'master')" >> ~/pe/.config/pe.jhbuildrc
echo "branches['lalmetaio'] = (None,'master')" >> ~/pe/.config/pe.jhbuildrc
echo "branches['lalxml'] = (None,'master')" >> ~/pe/.config/pe.jhbuildrc
echo "branches['lalburst'] = (None,'master')" >> ~/pe/.config/pe.jhbuildrc
echo "branches['lalpulsar'] = (None,'master')" >> ~/pe/.config/pe.jhbuildrc
echo "branches['lalstochastic'] = (None,'master')" >> ~/pe/.config/pe.jhbuildrc
echo "branches['lalinspiral'] = (None,'master')" >> ~/pe/.config/pe.jhbuildrc
echo "branches['lalsimulation'] = (None,'master')" >> ~/pe/.config/pe.jhbuildrc
echo "branches['lalinference'] = (None,'master')" >> ~/pe/.config/pe.jhbuildrc
echo "branches['lalapps'] = (None,'master')" >> ~/pe/.config/pe.jhbuildrc
echo "branches['glue'] = (None,'master')" >> ~/pe/.config/pe.jhbuildrc
echo "branches['pylal'] = (None,'master')" >> ~/pe/.config/pe.jhbuildrc
echo "branches['ligo'] = (None,'master')" >> ~/pe/.config/pe.jhbuildrc
echo "" >> ~/pe/.config/pe.jhbuildrc
echo "intel_executables = ['icc','icpc','ifort','mpiicc','mpiicpc','mpiifort','xiar']" >> ~/pe/.config/pe.jhbuildrc
echo "" >> ~/pe/.config/pe.jhbuildrc
echo "from distutils.spawn import find_executable" >> ~/pe/.config/pe.jhbuildrc
echo "def is_in_path(name):" >> ~/pe/.config/pe.jhbuildrc
echo '    """Check whether `name` is on PATH."""' >> ~/pe/.config/pe.jhbuildrc
echo "    return find_executable(name) is not None" >> ~/pe/.config/pe.jhbuildrc
echo "" >> ~/pe/.config/pe.jhbuildrc
echo "if all([is_in_path(name) for name in intel_executables]):" >> ~/pe/.config/pe.jhbuildrc
echo "   icc = True" >> ~/pe/.config/pe.jhbuildrc
echo "" >> ~/pe/.config/pe.jhbuildrc
echo "del name" >> ~/pe/.config/pe.jhbuildrc
echo "del intel_executables" >> ~/pe/.config/pe.jhbuildrc
echo "" >> ~/pe/.config/pe.jhbuildrc
echo "os.environ['LAL_DATA_PATH'] = '$HOME/ROM_data/'" >> ~/pe/.config/pe.jhbuildrc
echo "" >> ~/pe/.config/pe.jhbuildrc

# Install lalsuite from anonymous repository
~/pe/.local/bin/jhbuild -f ~/pe/.config/jhbuildrc --no-interact tinderbox --output=$HOME/public_html/pe/build/ --clean --distclean --force lalsuite

## If needed, install additional packages:
# ~/pe/.local/bin/jhbuild -f ~/pe/.config/jhbuildrc run $SHELL --noprofile --norc
# pip install <package> --install-option="--prefix=~/pe/local/"

# Create initisalisation script
echo '#!/bin/bash' > ~/pe/master.sh
echo "$HOME/pe/.local/bin/jhbuild -f $HOME/pe/.config/jhbuildrc run \$SHELL --noprofile --norc" >> ~/pe/master.sh
chmod a+x ~/pe/master.sh
